import torch

from physics_modular_3dgs.losses import photometric_invariant_structure_loss


def test_affine_exposure_is_cheaper_than_wrong_structure():
    axis = torch.linspace(-1.0, 1.0, 64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    pattern = 0.35 + 0.12 * torch.sin(9.0 * xx) + 0.10 * (yy > 0).float()
    image = torch.stack((pattern, pattern * 0.9, pattern * 0.8)).unsqueeze(0)
    exposure_changed = image * 0.65 + 0.13
    wrong_structure = torch.roll(image, shifts=7, dims=-1)
    weight = torch.ones(1, 1, 64, 64)

    exposure_loss = photometric_invariant_structure_loss(
        exposure_changed, image, weight, local_window=9
    )
    wrong_loss = photometric_invariant_structure_loss(
        wrong_structure, image, weight, local_window=9
    )
    assert torch.isfinite(exposure_loss)
    assert exposure_loss < 0.25 * wrong_loss


def test_structure_loss_backpropagates():
    prediction = torch.rand(1, 3, 24, 24, requires_grad=True)
    target = torch.rand_like(prediction)
    loss = photometric_invariant_structure_loss(
        prediction, target, torch.ones(1, 1, 24, 24), local_window=7
    )
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
