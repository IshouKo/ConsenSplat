from __future__ import annotations

import torch
import torch.nn.functional as F

from .types import LossWeights, PhysicsBatch, PhysicsOutput


def l1_optional(pred: torch.Tensor, target: torch.Tensor | None) -> torch.Tensor:
    if target is None:
        return pred.new_tensor(0.0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    if target.shape[-2:] != pred.shape[-2:]:
        target = F.interpolate(target, pred.shape[-2:], mode="bilinear", align_corners=False)
    return F.l1_loss(pred, target.to(device=pred.device, dtype=pred.dtype))


def total_variation(x: torch.Tensor) -> torch.Tensor:
    loss = x.new_tensor(0.0)
    if x.shape[-1] > 1:
        loss = loss + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    if x.shape[-2] > 1:
        loss = loss + (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    return loss


def _as_target(target: torch.Tensor | None, reference: torch.Tensor, mode: str = "bilinear"):
    if target is None:
        return None
    if target.dim() == 3:
        target = target.unsqueeze(0)
    target = target.to(device=reference.device, dtype=reference.dtype)
    if target.shape[-2:] != reference.shape[-2:]:
        kwargs = {} if mode == "nearest" else {"align_corners": False}
        target = F.interpolate(target, reference.shape[-2:], mode=mode, **kwargs)
    return target


def masked_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.expand_as(pred)
    return ((pred - target).abs() * weight).sum() / weight.sum().clamp_min(1.0)


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    padding = window_size // 2
    channel = pred.shape[1]
    kernel = pred.new_ones(channel, 1, window_size, window_size) / (window_size * window_size)
    mu_x = F.conv2d(pred, kernel, padding=padding, groups=channel)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=channel)
    sigma_x = F.conv2d(pred * pred, kernel, padding=padding, groups=channel) - mu_x * mu_x
    sigma_y = F.conv2d(target * target, kernel, padding=padding, groups=channel) - mu_y * mu_y
    sigma_xy = F.conv2d(pred * target, kernel, padding=padding, groups=channel) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return 1.0 - ssim.clamp(0.0, 1.0).mean()


def multiscale_ssim_loss(pred: torch.Tensor, target: torch.Tensor, levels: int = 3) -> torch.Tensor:
    losses = []
    for _ in range(max(1, int(levels))):
        losses.append(ssim_loss(pred, target))
        if min(pred.shape[-2:]) < 22:
            break
        pred = F.avg_pool2d(pred, 2, stride=2)
        target = F.avg_pool2d(target, 2, stride=2)
    return torch.stack(losses).mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.expand_as(pred)
    dx_weight = torch.minimum(weight[..., :, 1:], weight[..., :, :-1])
    dy_weight = torch.minimum(weight[..., 1:, :], weight[..., :-1, :])
    dx = ((pred[..., :, 1:] - pred[..., :, :-1]) - (target[..., :, 1:] - target[..., :, :-1])).abs()
    dy = ((pred[..., 1:, :] - pred[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])).abs()
    return (dx * dx_weight).sum() / dx_weight.sum().clamp_min(1.0) + (dy * dy_weight).sum() / dy_weight.sum().clamp_min(1.0)


def photometric_invariant_structure_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    local_window: int = 9,
) -> torch.Tensor:
    """Compare locally normalized gradient orientation and magnitude.

    Positive affine exposure changes are removed by local standardization, so
    the loss emphasizes persistent geometry/texture edges instead of the
    absolute RGB scale that is ambiguous with illumination and haze.
    """
    pred = pred if pred.ndim == 4 else pred.unsqueeze(0)
    target = target if target.ndim == 4 else target.unsqueeze(0)
    weight = weight if weight.ndim == 4 else weight.unsqueeze(0)
    if pred.shape[1] == 3:
        coefficients = pred.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        pred = (pred * coefficients).sum(dim=1, keepdim=True)
        target = (target * coefficients).sum(dim=1, keepdim=True)
    elif pred.shape[1] != 1:
        raise ValueError("photometric-invariant structure expects one or three channels")
    if weight.shape[1] != 1:
        weight = weight.mean(dim=1, keepdim=True)
    kernel = max(3, int(local_window) | 1)

    def normalize(image):
        mean = F.avg_pool2d(
            image, kernel, stride=1, padding=kernel // 2,
            count_include_pad=False,
        )
        second = F.avg_pool2d(
            image.square(), kernel, stride=1, padding=kernel // 2,
            count_include_pad=False,
        )
        deviation = (second - mean.square()).clamp_min(1e-4).sqrt()
        return (image - mean) / deviation

    pred = normalize(pred)
    target = normalize(target)
    sobel_x = pred.new_tensor([
        [-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0],
    ]).view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(-1, -2)
    pred_x = F.conv2d(pred, sobel_x, padding=1)
    pred_y = F.conv2d(pred, sobel_y, padding=1)
    target_x = F.conv2d(target, sobel_x, padding=1)
    target_y = F.conv2d(target, sobel_y, padding=1)
    pred_magnitude = (pred_x.square() + pred_y.square() + 1e-4).sqrt()
    target_magnitude = (target_x.square() + target_y.square() + 1e-4).sqrt()
    pred_direction = torch.cat((pred_x, pred_y), dim=1) / pred_magnitude
    target_direction = torch.cat((target_x, target_y), dim=1) / target_magnitude
    direction_error = (pred_direction - target_direction).abs().mean(dim=1, keepdim=True)
    magnitude_error = (pred_magnitude - target_magnitude).abs()
    edge_weight = (
        target_magnitude / target_magnitude.detach().mean().clamp_min(1e-3)
    ).clamp(0.25, 4.0)
    effective_weight = weight.clamp(0.0, 1.0) * edge_weight
    error = direction_error + 0.20 * magnitude_error
    return (error * effective_weight).sum() / effective_weight.sum().clamp_min(1.0)


def compute_physics_losses(
    output: PhysicsOutput,
    batch: PhysicsBatch,
    weights: LossWeights,
    prediction_weight: float = 1.0,
    particle_pixel_weight: float = 0.05,
    pseudo_clean_min_attenuation: float = 0.05,
    illumination_anchor: float = 0.8,
    clean_pseudo_scale: float = 1.0,
    perceptual_model=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    degraded = batch.degraded if batch.degraded.dim() == 4 else batch.degraded.unsqueeze(0)
    degraded = degraded.to(output.final.device, output.final.dtype)
    particle_mask = _as_target(batch.particle_mask_gt, output.final, "nearest")
    if particle_mask is None:
        particle_mask = output.particle_mask.detach()
    particle_mask = particle_mask[:, :1].clamp(0.0, 1.0)
    image_weight = 1.0 - (1.0 - float(particle_pixel_weight)) * particle_mask

    alpha = min(1.0, max(0.0, float(prediction_weight)))
    e_gt = _as_target(batch.effective_attenuation_gt, output.effective_attenuation)
    t_gt = _as_target(batch.transmission_gt, output.transmission)
    p_gt = _as_target(batch.path_radiance_gt, output.path_radiance)
    e_used = output.effective_attenuation if e_gt is None else alpha * output.effective_attenuation + (1.0 - alpha) * e_gt
    t_used = output.transmission if t_gt is None else alpha * output.transmission + (1.0 - alpha) * t_gt
    p_used = output.path_radiance if p_gt is None else alpha * output.path_radiance + (1.0 - alpha) * p_gt
    teacher_continuous = (output.clean_rgb * e_used * t_used + p_used).clamp(0.0, 1.0)

    terms: dict[str, torch.Tensor] = {}
    terms["image"] = masked_l1(teacher_continuous, degraded, image_weight)
    terms["low_light_attenuation"] = l1_optional(output.effective_attenuation, batch.effective_attenuation_gt)
    terms["transmission"] = l1_optional(output.transmission, batch.transmission_gt)
    terms["tau"] = l1_optional(output.optical_depth, batch.tau_gt)
    terms["path_radiance"] = l1_optional(output.path_radiance, batch.path_radiance_gt)
    explicit_clean = _as_target(batch.clean_pseudo_gt, output.clean_rgb)
    terms["consensus_ssim"] = output.final.new_tensor(0.0)
    terms["consensus_perceptual"] = output.final.new_tensor(0.0)
    terms["consensus_gradient"] = output.final.new_tensor(0.0)
    if explicit_clean is not None:
        confidence = _as_target(batch.clean_pseudo_confidence, output.clean_rgb)
        if confidence is None:
            confidence = torch.ones_like(explicit_clean[:, :1])
        confidence = confidence[:, :1] * (1.0 - particle_mask)
        pseudo_target = explicit_clean.detach()
        pseudo_weight = confidence.detach()
        terms["clean_pseudo"] = masked_l1(output.clean_rgb, pseudo_target, pseudo_weight)
        masked_pred = output.clean_rgb * pseudo_weight
        masked_target = pseudo_target * pseudo_weight
        if weights.consensus_ssim:
            terms["consensus_ssim"] = multiscale_ssim_loss(masked_pred, masked_target)
        if weights.consensus_gradient:
            terms["consensus_gradient"] = gradient_loss(output.clean_rgb, pseudo_target, pseudo_weight)
        if weights.consensus_perceptual and perceptual_model is not None:
            max_side = max(masked_pred.shape[-2:])
            if max_side > 256:
                scale = 256.0 / max_side
                size = tuple(max(16, round(value * scale)) for value in masked_pred.shape[-2:])
                masked_pred = F.interpolate(masked_pred, size, mode="bilinear", align_corners=False)
                masked_target = F.interpolate(masked_target, size, mode="bilinear", align_corners=False)
            terms["consensus_perceptual"] = perceptual_model(masked_pred, masked_target).mean()
    elif e_gt is not None and t_gt is not None and p_gt is not None:
        denom = e_gt * t_gt
        pseudo_clean = ((degraded - p_gt) / denom.clamp_min(1e-4)).clamp(0.0, 1.0).detach()
        confidence = ((denom >= float(pseudo_clean_min_attenuation)).to(denom.dtype) * (1.0 - particle_mask)).detach()
        terms["clean_pseudo"] = masked_l1(output.clean_rgb, pseudo_clean, confidence)
    else:
        terms["clean_pseudo"] = output.final.new_tensor(0.0)
    terms["illumination_anchor"] = (output.effective_attenuation.mean() - float(illumination_anchor)).abs()
    smooth_field = output.base_illumination if output.base_illumination is not None else output.effective_attenuation
    terms["low_light_smooth"] = total_variation(smooth_field)
    terms["depth_tv"] = total_variation(output.depth)
    terms["ssim"] = ssim_loss(teacher_continuous * image_weight, degraded * image_weight) if weights.ssim else output.final.new_tensor(0.0)
    if batch.beta_gt is not None:
        beta_gt = batch.beta_gt.to(output.final.device, output.final.dtype).mean()
        terms["beta"] = (output.beta - beta_gt).abs()
    else:
        terms["beta"] = output.final.new_tensor(0.0)
    if batch.airlight_gt is not None:
        air = batch.airlight_gt.to(output.path_radiance.device, output.path_radiance.dtype).reshape(-1, 3)
        terms["airlight"] = (output.airlight.view(1, 3) - air).abs().mean()
    else:
        terms["airlight"] = output.final.new_tensor(0.0)

    total = output.final.new_tensor(0.0)
    for name, value in terms.items():
        coefficient = float(getattr(weights, name, weights.extras.get(name, 0.0)))
        if name in {"clean_pseudo", "consensus_ssim", "consensus_perceptual", "consensus_gradient"}:
            coefficient *= min(1.0, max(0.0, float(clean_pseudo_scale)))
        total = total + coefficient * value
    return total, terms
