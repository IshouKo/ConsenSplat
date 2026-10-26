import unittest

import torch

from physics_modular_3dgs.model import PhysicsAware3DGS
from physics_modular_3dgs.losses import compute_physics_losses
from physics_modular_3dgs.types import LossWeights, PhysicsBatch, RasterOutputs


def make_inputs() -> tuple[RasterOutputs, torch.Tensor]:
    clean = torch.full((1, 3, 8, 8), 0.25)
    depth = torch.ones(1, 1, 8, 8)
    normal = torch.zeros(1, 3, 8, 8)
    normal[:, 2] = 1.0
    features = torch.zeros(1, 8, 8, 8)
    degraded = torch.full((1, 3, 8, 8), 0.75)
    return RasterOutputs(clean, depth, normal, features, features.clone()), degraded


class ParticleModelTest(unittest.TestCase):
    def test_low_light_attenuation_is_direct_single_channel_output(self):
        raster, degraded = make_inputs()
        model = PhysicsAware3DGS(num_views=1)

        output = model(raster, degraded, particle_enabled=False)

        self.assertEqual(output.effective_attenuation.shape, (1, 1, 8, 8))
        self.assertTrue(bool((output.effective_attenuation >= 0.0).all()))
        self.assertTrue(bool((output.effective_attenuation <= 1.0).all()))
        torch.testing.assert_close(
            output.low_light_clean,
            output.clean_rgb * output.effective_attenuation,
        )
        self.assertFalse(hasattr(output, "exposure_gamma"))

    def test_effective_attenuation_supervision_reaches_decoder(self):
        raster, degraded = make_inputs()
        model = PhysicsAware3DGS(num_views=1)
        output = model(raster, degraded, particle_enabled=False)
        batch = PhysicsBatch(
            degraded=degraded,
            effective_attenuation_gt=torch.full((1, 1, 8, 8), 0.25),
        )

        loss, terms = compute_physics_losses(
            output,
            batch,
            LossWeights(
                image=0.0,
                low_light_attenuation=1.0,
                transmission=0.0,
                path_radiance=0.0,
                depth_tv=0.0,
                low_light_smooth=0.0,
            ),
        )
        loss.backward()

        self.assertNotIn("clean", terms)
        self.assertGreater(float(terms["low_light_attenuation"]), 0.0)
        grads = [p.grad for p in model.low_light_decoder.parameters()]
        self.assertTrue(any(g is not None and bool((g != 0).any()) for g in grads))

    def test_disabled_particle_layer_is_exactly_zero(self):
        raster, degraded = make_inputs()
        model = PhysicsAware3DGS(num_views=1)

        output = model(raster, degraded, particle_enabled=False)

        torch.testing.assert_close(output.final, output.continuous_degraded)
        self.assertEqual(float(output.particle_mask.max()), 0.0)
        self.assertEqual(float(output.particle_layer.max()), 0.0)

    def test_low_only_mode_hard_disables_haze_and_particles(self):
        raster, degraded = make_inputs()
        model = PhysicsAware3DGS(
            num_views=1,
            haze_decomposition_enabled=False,
            particle_decomposition_enabled=False,
        )

        output = model(
            raster,
            degraded,
            particle_enabled=True,
            particle_residual=torch.full_like(degraded, 0.2),
        )

        torch.testing.assert_close(output.transmission, torch.ones_like(output.transmission))
        torch.testing.assert_close(output.path_radiance, torch.zeros_like(output.path_radiance))
        torch.testing.assert_close(output.particle_layer, torch.zeros_like(output.particle_layer))
        torch.testing.assert_close(output.continuous_degraded, output.low_light_clean)
        torch.testing.assert_close(output.final, output.low_light_clean)

    def test_enabled_particle_buffer_is_additive_and_nonnegative(self):
        raster, degraded = make_inputs()
        model = PhysicsAware3DGS(num_views=1)

        residual = torch.full_like(degraded, 0.2)
        output = model(
            raster, degraded, particle_enabled=True, particle_strength=0.5,
            particle_residual=residual,
        )

        self.assertTrue(bool((output.particle_residual >= 0.0).all()))
        self.assertTrue(bool((output.particle_layer >= 0.0).all()))
        expected = (output.continuous_degraded + 0.5 * output.particle_layer).clamp(0.0, 1.0)
        torch.testing.assert_close(output.final, expected)

    def test_missing_buffer_does_not_leak_the_observed_target(self):
        raster, degraded = make_inputs()
        model = PhysicsAware3DGS(num_views=1)

        output = model(raster, degraded, particle_enabled=True)

        torch.testing.assert_close(output.final, output.continuous_degraded)
        self.assertEqual(float(output.particle_residual.max()), 0.0)

    def test_particle_residual_is_a_detached_non_learned_buffer(self):
        raster, degraded = make_inputs()
        model = PhysicsAware3DGS(num_views=1)
        residual = torch.full_like(degraded, 0.2, requires_grad=True)

        output = model(
            raster, degraded, particle_enabled=True, particle_residual=residual
        )

        self.assertFalse(output.particle_residual.requires_grad)
        self.assertFalse(any("particle" in name for name, _ in model.named_parameters()))
        torch.testing.assert_close(output.particle_layer, residual.detach())


    def test_particle_residual_is_zero_outside_gt_mask(self):
        raster, degraded = make_inputs()
        model = PhysicsAware3DGS(num_views=1)
        residual = torch.full_like(degraded, 0.2)
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., :4, :4] = 1.0
        output = model(raster, degraded, particle_residual=residual, particle_mask_override=mask)
        self.assertEqual(float(output.particle_layer[..., 4:, 4:].max()), 0.0)

    def test_pseudo_clean_recovers_clean_on_valid_nonparticle_pixels(self):
        raster, _ = make_inputs()
        model = PhysicsAware3DGS(num_views=1)
        clean = raster.clean_rgb.detach()
        e = torch.full((1, 1, 8, 8), 0.5)
        t = torch.full((1, 1, 8, 8), 0.8)
        p = torch.full((1, 3, 8, 8), 0.1)
        degraded = clean * e * t + p
        output = model(raster, degraded, particle_enabled=False)
        batch = PhysicsBatch(degraded=degraded, effective_attenuation_gt=e, transmission_gt=t, path_radiance_gt=p, particle_mask_gt=torch.zeros(1, 1, 8, 8))
        _, terms = compute_physics_losses(output, batch, LossWeights(clean_pseudo=1.0))
        self.assertLess(float(terms["clean_pseudo"]), 1e-6)


if __name__ == "__main__":
    unittest.main()
