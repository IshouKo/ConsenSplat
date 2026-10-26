from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from .model import PhysicsAware3DGS
from .types import PhysicsBatch, PhysicsOutput, RasterOutputs


def inverse_depth_to_distance(inverse_depth: torch.Tensor) -> torch.Tensor:
    """Convert the 3DGS rasterizer inverse-depth output to a finite path length."""
    valid = inverse_depth > 1e-6
    distance = torch.zeros_like(inverse_depth)
    distance[valid] = inverse_depth[valid].reciprocal()
    if valid.any():
        far = distance[valid].detach().quantile(0.99).clamp_min(1e-3)
        distance = torch.where(valid, distance.clamp_max(far), far)
    else:
        distance.fill_(1.0)
    return distance


def normals_from_depth(depth: torch.Tensor) -> torch.Tensor:
    depth = depth if depth.dim() == 4 else depth.unsqueeze(0)
    dx = F.pad(depth[..., 1:] - depth[..., :-1], (0, 1, 0, 0))
    dy = F.pad(depth[..., 1:, :] - depth[..., :-1, :], (0, 0, 0, 1))
    normal = torch.cat((-dx, -dy, torch.ones_like(depth)), dim=1)
    return F.normalize(normal, dim=1, eps=1e-6)


def _feature_bank(clean: torch.Tensor, depth: torch.Tensor, normal: torch.Tensor, channels: int) -> torch.Tensor:
    depth_scale = depth.detach().flatten(1).quantile(0.95, dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
    depth_norm = (depth / depth_scale).clamp(0.0, 1.0)
    luminance = (0.2126 * clean[:, :1] + 0.7152 * clean[:, 1:2] + 0.0722 * clean[:, 2:3])
    base = torch.cat((clean, depth_norm, normal, luminance), dim=1)
    repeats = (channels + base.shape[1] - 1) // base.shape[1]
    return base.repeat(1, repeats, 1, 1)[:, :channels]


def raster_outputs_from_3dgs(render_pkg: dict[str, torch.Tensor], feature_channels: int = 8) -> RasterOutputs:
    clean = render_pkg["render"]
    clean = clean if clean.dim() == 4 else clean.unsqueeze(0)
    inverse_depth = render_pkg["depth"]
    inverse_depth = inverse_depth if inverse_depth.dim() == 4 else inverse_depth.unsqueeze(0)
    depth = inverse_depth_to_distance(inverse_depth[:, :1])
    normal = normals_from_depth(depth)
    fallback_features = _feature_bank(clean, depth, normal, feature_channels)
    illum_features = render_pkg.get("illum_features", fallback_features)
    degradation_features = render_pkg.get("degradation_features", fallback_features)
    illum_features = illum_features if illum_features.dim() == 4 else illum_features.unsqueeze(0)
    degradation_features = degradation_features if degradation_features.dim() == 4 else degradation_features.unsqueeze(0)
    return RasterOutputs(
        clean_rgb=clean,
        depth=depth,
        normal=normal,
        illum_features=illum_features,
        degradation_features=degradation_features,
    )


def run_physics(
    model: PhysicsAware3DGS,
    render_pkg: dict[str, torch.Tensor],
    observed: torch.Tensor,
    view_index: int,
    feature_channels: int = 8,
    particle_enabled: bool = True,
    particle_strength: float = 1.0,
    particle_residual: torch.Tensor | None = None,
    particle_mask: torch.Tensor | None = None,
    low_light_trust_weight: float = 1.0,
    transmission_trust_weight: float = 1.0,
    particle_trust_weight: float = 1.0,
    camera: Any | None = None,
) -> tuple[PhysicsOutput, RasterOutputs]:
    raster = raster_outputs_from_3dgs(render_pkg, feature_channels)
    observed = observed if observed.dim() == 4 else observed.unsqueeze(0)
    model_device = raster.clean_rgb.device
    observed = observed.to(device=model_device, dtype=raster.clean_rgb.dtype)
    index = torch.tensor([view_index], dtype=torch.long, device=model_device)
    return model(
        raster,
        observed,
        index,
        particle_enabled=particle_enabled,
        particle_strength=particle_strength,
        particle_residual=particle_residual,
        particle_mask_override=particle_mask,
        low_light_trust_weight=low_light_trust_weight,
        transmission_trust_weight=transmission_trust_weight,
        particle_trust_weight=particle_trust_weight,
        camera=camera,
    ), raster


def project_reliability_to_gaussians(
    camera: Any,
    xyz: torch.Tensor,
    reliability: torch.Tensor,
    radii: torch.Tensor,
    floor: float = 0.0,
    invisible_value: float = 1.0,
) -> torch.Tensor:
    """Sample radius-selected 3/7/15 AvgPool trust at projected centers."""
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    homogeneous = torch.cat((xyz, ones), dim=1)
    clip = homogeneous @ camera.full_proj_transform
    w = clip[:, 3]
    safe_w = torch.where(w.abs() > 1e-6, w, torch.ones_like(w))
    grid = torch.stack((clip[:, 0] / safe_w, clip[:, 1] / safe_w), dim=-1)
    trust = reliability.detach()
    pooled = [F.avg_pool2d(trust, k, stride=1, padding=k // 2) for k in (3, 7, 15)]
    samples = [
        F.grid_sample(
            level, grid.view(1, -1, 1, 2), mode="bilinear",
            padding_mode="zeros", align_corners=True,
        ).view(-1)
        for level in pooled
    ]
    radii = radii.detach().to(device=xyz.device).view(-1)
    if radii.numel() != xyz.shape[0]:
        raise ValueError("Projected radii count must match the number of Gaussians.")
    sampled = torch.where(radii < 2.0, samples[0], torch.where(radii < 6.0, samples[1], samples[2]))
    visible = (w > 0) & (grid.abs() <= 1.0).all(dim=1)
    invisible = torch.full_like(sampled, float(invisible_value))
    return torch.where(visible, sampled.clamp(floor, 1.0), invisible)


def load_optional_reference(
    root: str | Path | None,
    image_name: str,
    size: tuple[int, int],
    device: torch.device,
    grayscale: bool = False,
):
    if not root:
        return None
    root = Path(root)
    stem = Path(image_name).stem
    candidates = [root / image_name]
    candidates.extend(root / f"{stem}{suffix}" for suffix in (".png", ".jpg", ".jpeg", ".JPG", ".PNG"))
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return None
    image = pil_to_tensor(
        Image.open(path).convert("L" if grayscale else "RGB")
    ).float().div_(255.0).unsqueeze(0).to(device)
    if image.shape[-2:] != size:
        image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
    return image


def save_physics_state(
    path: str | Path,
    model: PhysicsAware3DGS,
    iteration: int,
    feature_channels: int,
    optimizer: torch.optim.Optimizer | None = None,
    particle_buffers: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration": int(iteration),
        "feature_channels": int(feature_channels),
        "num_views": int(model.view_condition.num_embeddings),
        "model": model.state_dict(),
        "model_config": {
            "physics_v2": model.physics_v2,
            "low_light_decomposition_enabled": model.low_light_decomposition_enabled,
            "haze_decomposition_enabled": model.haze_decomposition_enabled,
            "particle_decomposition_enabled": model.particle_decomposition_enabled,
            "gamma_min": model.gamma_min,
            "gamma_max": model.gamma_max,
            "haze_depth_normalized": model.medium.normalized_depth,
            "haze_depth_blur_kernel": model.medium.depth_blur_kernel,
            "haze_depth_max": model.medium.depth_max,
            "illumination_grid_size": model.low_light_decoder.grid_size,
        },
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if particle_buffers is not None:
        payload["particle_buffers"] = particle_buffers
    torch.save(payload, path)


def load_physics_state(path: str | Path, device: torch.device) -> tuple[PhysicsAware3DGS, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    config = payload.get("model_config", {})
    model = PhysicsAware3DGS(
        illum_feature_channels=int(payload.get("feature_channels", 8)),
        degradation_feature_channels=int(payload.get("feature_channels", 8)),
        num_views=int(payload.get("num_views", 1)),
        physics_v2=bool(config.get("physics_v2", False)),
        low_light_decomposition_enabled=bool(config.get("low_light_decomposition_enabled", True)),
        haze_decomposition_enabled=bool(config.get("haze_decomposition_enabled", True)),
        particle_decomposition_enabled=bool(config.get("particle_decomposition_enabled", True)),
        gamma_min=float(config.get("gamma_min", 1.4)),
        gamma_max=float(config.get("gamma_max", 1.9)),
        haze_depth_normalized=bool(config.get("haze_depth_normalized", False)),
        haze_depth_blur_kernel=int(config.get("haze_depth_blur_kernel", 21)),
        haze_depth_max=float(config.get("haze_depth_max", 0.7)),
        illumination_grid_size=int(config.get("illumination_grid_size", 16)),
    ).to(device)
    # Particle Encoder/Mask Head checkpoints from the earlier prototype may
    # contain now-obsolete keys; the physical branches remain loadable.
    model.load_state_dict(payload["model"], strict=False)
    return model, payload


def write_run_metadata(model_path: str | Path, values: dict[str, Any]) -> None:
    path = Path(model_path) / "physics_config.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2)


def make_training_batch(
    observed: torch.Tensor,
    depth: torch.Tensor,
    effective_attenuation_gt: torch.Tensor | None = None,
    transmission_gt: torch.Tensor | None = None,
    tau_gt: torch.Tensor | None = None,
    path_radiance_gt: torch.Tensor | None = None,
    clean_pseudo_gt: torch.Tensor | None = None,
    clean_pseudo_confidence: torch.Tensor | None = None,
    particle_mask_gt: torch.Tensor | None = None,
    beta_gt: torch.Tensor | None = None,
    airlight_gt: torch.Tensor | None = None,
) -> PhysicsBatch:
    observed = observed if observed.dim() == 4 else observed.unsqueeze(0)
    return PhysicsBatch(
        degraded=observed,
        depth_gt=depth,
        effective_attenuation_gt=effective_attenuation_gt,
        transmission_gt=transmission_gt,
        tau_gt=tau_gt,
        path_radiance_gt=path_radiance_gt,
        clean_pseudo_gt=clean_pseudo_gt,
        clean_pseudo_confidence=clean_pseudo_confidence,
        particle_mask_gt=particle_mask_gt,
        beta_gt=beta_gt,
        airlight_gt=airlight_gt,
    )
