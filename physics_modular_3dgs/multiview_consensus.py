from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass
class ConsensusLabel:
    clean: torch.Tensor
    confidence: torch.Tensor
    iteration: int


def invert_degradation(observed, attenuation, transmission, path_radiance, eps=1e-3):
    """Invert the hard image-formation model using predicted physical factors."""
    observed = observed if observed.dim() == 4 else observed.unsqueeze(0)
    denom = (attenuation * transmission).clamp_min(float(eps))
    return ((observed.to(denom) - path_radiance) / denom).clamp(0.0, 1.0)


def _view_direction(camera):
    rotation = torch.as_tensor(camera.R, dtype=torch.float32)
    return torch.nn.functional.normalize(rotation[:, 2], dim=0)


def nearest_camera_map(cameras, neighbors=2, candidates=8):
    """Select nearby cameras while rejecting redundant viewing directions."""
    result = {}
    for target in cameras:
        nearby = sorted(
            (source for source in cameras if source is not target),
            key=lambda source: float((source.camera_center - target.camera_center).square().sum()),
        )[: max(int(neighbors), int(candidates))]
        if not nearby:
            result[target.image_name] = []
            continue
        distances = torch.tensor([
            float((source.camera_center - target.camera_center).norm()) for source in nearby
        ])
        distance_scale = distances.median().clamp_min(1e-6)
        target_direction = _view_direction(target)
        ranked = sorted(
            zip(nearby, distances),
            key=lambda item: (
                float(item[1] / distance_scale)
                + 0.5 * (1.0 - float(torch.dot(target_direction, _view_direction(item[0])).clamp(-1, 1)))
            ),
        )
        result[target.image_name] = [camera for camera, _ in ranked[: max(1, int(neighbors))]]
    return result


def _target_world_points(camera, depth):
    depth = depth if depth.dim() == 4 else depth.unsqueeze(0)
    z = depth[0, 0]
    h, w = z.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=z.device, dtype=z.dtype),
        torch.arange(w, device=z.device, dtype=z.dtype),
        indexing="ij",
    )
    fx = w / (2.0 * math.tan(float(camera.FoVx) * 0.5))
    fy = h / (2.0 * math.tan(float(camera.FoVy) * 0.5))
    x = (xx - (w - 1) * 0.5) * z / fx
    y = (yy - (h - 1) * 0.5) * z / fy
    camera_points = torch.stack((x, y, z, torch.ones_like(z)), dim=-1).reshape(-1, 4)
    return camera_points @ torch.linalg.inv(camera.world_view_transform.to(z))


def reproject_consensus(
    target_camera, target_depth, sources, agreement_scale=0.10,
    depth_consistency=0.10, min_views=2, confidence_threshold=0.50,
):
    """Warp source clean estimates to a target view and take a robust median."""
    if not sources:
        raise ValueError("At least one source camera/estimate pair is required")
    world = _target_world_points(target_camera, target_depth)
    h, w = target_depth.shape[-2:]
    warped, valid_masks = [], []
    for source_camera, source_clean, source_depth in sources:
        clip = world @ source_camera.full_proj_transform.to(world)
        clip_w = clip[:, 3]
        safe_w = torch.where(clip_w.abs() > 1e-6, clip_w, torch.ones_like(clip_w))
        grid = (clip[:, :2] / safe_w[:, None]).reshape(1, h, w, 2)
        valid = ((clip_w > 0) & (grid.reshape(-1, 2).abs() <= 1).all(dim=1)).reshape(1, 1, h, w)
        source_z = (world @ source_camera.world_view_transform.to(world))[:, 2].reshape(1, 1, h, w)
        source_depth = source_depth if source_depth.dim() == 4 else source_depth.unsqueeze(0)
        sampled_depth = F.grid_sample(
            source_depth.to(world), grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        relative_depth_error = (sampled_depth - source_z).abs() / sampled_depth.abs().clamp_min(1e-3)
        valid = valid & (sampled_depth > 1e-6) & (source_z > 0) & (relative_depth_error <= float(depth_consistency))
        image = source_clean if source_clean.dim() == 4 else source_clean.unsqueeze(0)
        sample = F.grid_sample(
            image.to(world), grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        warped.append(sample)
        valid_masks.append(valid)
    values = torch.stack(warped, dim=0)
    valid = torch.stack(valid_masks, dim=0)
    values_nan = values.masked_fill(~valid.expand_as(values), float("nan"))
    clean = torch.nanmedian(values_nan, dim=0).values
    clean = torch.nan_to_num(clean, nan=0.0).clamp(0.0, 1.0)
    deviation = (values - clean.unsqueeze(0)).abs().mean(dim=2, keepdim=True)
    deviation = deviation.masked_fill(~valid, float("nan"))
    mad = torch.nanmedian(deviation, dim=0).values
    count = valid.float().sum(dim=0)
    confidence = torch.exp(
        -torch.nan_to_num(mad, nan=1.0) / max(float(agreement_scale), 1e-4)
    )
    confidence = confidence * (count / len(sources)).clamp(0.0, 1.0)
    confidence = confidence * (count >= int(min_views)).to(confidence.dtype)
    confidence = torch.where(confidence >= float(confidence_threshold), confidence, torch.zeros_like(confidence))
    return clean.detach(), confidence.detach()


class ConsensusCache:
    def __init__(self, refresh_interval=200, ema_decay=0.0):
        self.refresh_interval = max(1, int(refresh_interval))
        self.ema_decay = min(max(float(ema_decay), 0.0), 0.9999)
        self._labels = {}

    def needs_refresh(self, key, iteration):
        value = self._labels.get(key)
        return value is None or int(iteration) - value.iteration >= self.refresh_interval

    def get(self, key, device, dtype):
        value = self._labels.get(key)
        if value is None:
            return None
        return ConsensusLabel(
            value.clean.to(device=device, dtype=dtype),
            value.confidence.to(device=device, dtype=dtype),
            value.iteration,
        )

    def put(self, key, clean, confidence, iteration):
        previous = self._labels.get(key)
        if previous is not None and self.ema_decay > 0.0:
            old_clean = previous.clean.to(device=clean.device, dtype=clean.dtype)
            old_confidence = previous.confidence.to(device=confidence.device, dtype=confidence.dtype)
            decay = self.ema_decay
            clean = decay * old_clean + (1.0 - decay) * clean
            confidence = decay * old_confidence + (1.0 - decay) * confidence
        self._labels[key] = ConsensusLabel(
            clean.detach().cpu(), confidence.detach().cpu(), int(iteration)
        )
        return self.get(key, clean.device, clean.dtype)
