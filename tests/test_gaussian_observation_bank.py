import torch

from physics_modular_3dgs.gaussian_observation_bank import (
    GaussianObservationBank, calibrate_target_brightness, gauge_fix_rgb_update,
    residual_base_targets, robust_luminance_quantile,
)
from utils.sh_utils import RGB2SH


def test_quantile_keeps_rgb_triplet_and_rejects_particle_outlier():
    values = torch.tensor([[[.30, .35, .20], [.42, .48, .28], [.40, .46, .27], [1., 1., 1.], [.38, .44, .25]]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    target, confidence, usable = robust_luminance_quantile(values, valid, .8, 3, 2.5)
    assert usable.item()
    assert not torch.allclose(target[0], torch.ones(3))
    assert any(torch.allclose(target[0], row) for row in values[0, :3].unbind(0)) or torch.allclose(target[0], values[0, 4])
    assert confidence.item() > 0


def test_low_view_count_disables_target_and_loss_is_finite():
    torch.manual_seed(0)
    values = torch.rand(2, 4, 3)
    valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    bank = GaussianObservationBank(min_views=3)
    targets = bank.build(values, valid, 10)
    assert not targets.valid[0]
    assert targets.valid[1]
    features = RGB2SH(torch.full((2, 3), .5)).unsqueeze(1).requires_grad_()
    loss = bank.loss(features)
    assert torch.isfinite(loss)
    loss.backward()
    assert features.grad is not None


def test_brightness_calibration_anchors_median_with_bounded_gain():
    target = torch.full((64, 3), 0.40)
    reference = torch.full((64, 3), 0.60)
    confidence = torch.ones(64)
    usable = torch.ones(64, dtype=torch.bool)
    calibrated, gain = calibrate_target_brightness(
        target, reference, confidence, usable, gain_limit=0.15
    )
    assert abs(gain - 1.15) < 1e-6
    assert torch.allclose(calibrated, torch.full_like(calibrated, 0.46))


def test_bank_requires_reference_when_brightness_calibration_is_enabled():
    bank = GaussianObservationBank(min_views=2, brightness_calibration=True)
    observations = torch.rand(64, 3, 3)
    valid = torch.ones(64, 3, dtype=torch.bool)
    try:
        bank.build(observations, valid, 1)
    except ValueError as error:
        assert "reference_rgb" in str(error)
    else:
        raise AssertionError("missing brightness reference should fail")


def test_residual_base_target_transfers_bounded_render_residual():
    corrected = torch.tensor([[0.80, 0.30, 0.20], [0.10, 0.90, 0.40]])
    rendered = torch.tensor([[0.50, 0.35, 0.25], [0.30, 0.60, 0.40]])
    base = torch.tensor([[0.40, 0.50, 0.60], [0.20, 0.30, 0.40]])
    target = residual_base_targets(corrected, rendered, base, max_delta=0.10)
    expected = torch.tensor([[0.50, 0.45, 0.55], [0.10, 0.40, 0.40]])
    assert torch.allclose(target, expected)


def test_gauge_fix_removes_weighted_global_rgb_update():
    reference = torch.full((4, 3), 0.40)
    target = reference + torch.tensor([
        [.10, .04, .02], [.06, .08, .02], [.02, .02, .08], [.20, .20, .20],
    ])
    confidence = torch.tensor([1.0, 0.5, 1.0, 1.0])
    usable = torch.tensor([True, True, True, False])
    fixed, offset = gauge_fix_rgb_update(target, reference, confidence, usable)
    residual = fixed - reference
    weighted_mean = (residual * (confidence * usable)[:, None]).sum(0) / (confidence * usable).sum()
    assert torch.allclose(weighted_mean, torch.zeros(3), atol=1e-6)
    assert torch.allclose(fixed[3], target[3])
    assert torch.all(offset > 0)
