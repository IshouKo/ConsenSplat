from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class RasterOutputs:
    clean_rgb: torch.Tensor
    depth: torch.Tensor
    normal: torch.Tensor
    illum_features: torch.Tensor
    degradation_features: torch.Tensor
    accumulated_alpha: Optional[torch.Tensor] = None


@dataclass
class PhysicsBatch:
    degraded: torch.Tensor
    depth_gt: Optional[torch.Tensor] = None
    effective_attenuation_gt: Optional[torch.Tensor] = None
    transmission_gt: Optional[torch.Tensor] = None
    tau_gt: Optional[torch.Tensor] = None
    path_radiance_gt: Optional[torch.Tensor] = None
    particle_mask_gt: Optional[torch.Tensor] = None
    particle_residual_gt: Optional[torch.Tensor] = None
    clean_pseudo_gt: Optional[torch.Tensor] = None
    clean_pseudo_confidence: Optional[torch.Tensor] = None
    beta_gt: Optional[torch.Tensor] = None
    airlight_gt: Optional[torch.Tensor] = None
    view_index: Optional[torch.Tensor] = None


@dataclass
class PhysicsOutput:
    clean_rgb: torch.Tensor
    low_light_clean: torch.Tensor
    effective_attenuation: torch.Tensor
    transmission: torch.Tensor
    optical_depth: torch.Tensor
    path_radiance: torch.Tensor
    continuous_degraded: torch.Tensor
    residual_abs: torch.Tensor
    residual_pos: torch.Tensor
    residual_neg: torch.Tensor
    particle_mask: torch.Tensor
    particle_residual: torch.Tensor
    particle_layer: torch.Tensor
    final: torch.Tensor
    pixel_reliability: torch.Tensor
    depth: torch.Tensor
    beta: torch.Tensor
    airlight: torch.Tensor
    base_illumination: Optional[torch.Tensor] = None
    gamma: Optional[torch.Tensor] = None
    haze_depth: Optional[torch.Tensor] = None


@dataclass
class LossWeights:
    image: float = 1.0
    low_light_attenuation: float = 0.1
    transmission: float = 0.5
    tau: float = 0.0
    path_radiance: float = 0.1
    clean_pseudo: float = 0.0
    illumination_anchor: float = 0.0
    consensus_ssim: float = 0.0
    consensus_perceptual: float = 0.0
    consensus_gradient: float = 0.0
    beta: float = 0.0
    airlight: float = 0.0
    depth_tv: float = 0.001
    low_light_smooth: float = 0.02
    ssim: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)
