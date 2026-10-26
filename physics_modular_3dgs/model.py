from __future__ import annotations

import torch
from torch import nn

from .modules import HomogeneousMedium, LowLightAttenuationDecoder
from .reliability import pixel_reliability
from .types import PhysicsOutput, RasterOutputs
from fog_field import FogTransmissionField
from airlight_field import AirlightField
from nimbusgs.rendering import synthesize_fogged_image


class PhysicsAware3DGS(nn.Module):
    def __init__(
        self,
        illum_feature_channels: int = 8,
        degradation_feature_channels: int = 8,
        num_views: int = 1,
        view_condition_dim: int = 8,
        beta_init: float = 0.08,
        airlight_init: tuple[float, float, float] = (0.8, 0.8, 0.8),
        illumination_grid_size: int = 16,
        illumination_mode: str = "decoder",
        physics_v2: bool = False,
        gamma_min: float = 1.4,
        gamma_max: float = 1.9,
        haze_depth_normalized: bool = False,
        haze_depth_blur_kernel: int = 21,
        haze_depth_max: float = 0.7,
        csm_enabled: bool = False,
        csm_gaussians=None,
        csm_resolution: int = 128,
        csm_num_samples: int = 64,
        csm_density_scale_init: float = 0.3,
        csm_airlight_mode: str = "cnn",
        low_light_decomposition_enabled: bool = True,
        haze_decomposition_enabled: bool = True,
        particle_decomposition_enabled: bool = True,
    ):
        super().__init__()
        self.physics_v2 = bool(physics_v2)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.low_light_decomposition_enabled = bool(low_light_decomposition_enabled)
        self.haze_decomposition_enabled = bool(haze_decomposition_enabled)
        self.particle_decomposition_enabled = bool(particle_decomposition_enabled)
        self.view_condition = nn.Embedding(num_views, view_condition_dim)
        self.low_light_decoder = LowLightAttenuationDecoder(
            illum_feature_channels,
            view_condition_dim=view_condition_dim,
            grid_size=illumination_grid_size,
            mode=illumination_mode,
        )
        self.medium = HomogeneousMedium(
            beta_init=beta_init,
            airlight_init=airlight_init,
            normalized_depth=haze_depth_normalized,
            depth_blur_kernel=haze_depth_blur_kernel,
            depth_max=haze_depth_max,
        )
        self.csm_enabled = bool(csm_enabled)
        if self.csm_enabled and not self.haze_decomposition_enabled:
            raise ValueError("CSM cannot be enabled when haze decomposition is disabled")
        self.csm_num_samples = int(csm_num_samples)
        if self.csm_enabled:
            self.csm_fog = FogTransmissionField(
                gaussians=csm_gaussians, resolution=int(csm_resolution),
                density_scale_init=float(csm_density_scale_init), auto_normalize=True,
            )
            self.csm_airlight = AirlightField(mode=str(csm_airlight_mode))
        # Keep V1's original random-initialization order intact.  V2-only
        # parameters must be initialized after the V1 modules.
        self.gamma_logit = nn.Embedding(num_views, 1)
        nn.init.zeros_(self.gamma_logit.weight)

    def forward(
        self,
        raster: RasterOutputs,
        degraded: torch.Tensor,
        view_index: torch.Tensor | None = None,
        particle_enabled: bool = True,
        particle_strength: float = 1.0,
        particle_residual: torch.Tensor | None = None,
        particle_mask_override: torch.Tensor | None = None,
        low_light_trust_weight: float = 1.0,
        transmission_trust_weight: float = 1.0,
        particle_trust_weight: float = 1.0,
        camera=None,
    ) -> PhysicsOutput:
        if degraded.dim() == 3:
            degraded = degraded.unsqueeze(0)
        b = degraded.shape[0]
        if view_index is None:
            view_index = torch.zeros(b, dtype=torch.long, device=degraded.device)
        clean = raster.clean_rgb if raster.clean_rgb.dim() == 4 else raster.clean_rgb.unsqueeze(0)
        if self.low_light_decomposition_enabled:
            view_condition = self.view_condition(view_index.long().view(-1))
            base_illumination = self.low_light_decoder(
                raster.illum_features,
                raster.depth,
                raster.normal,
                view_condition,
            )
        else:
            base_illumination = torch.ones_like(clean[:, :1])
        if self.physics_v2 and self.low_light_decomposition_enabled:
            gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(
                self.gamma_logit(view_index.long().view(-1))
            )
            effective_attenuation = (base_illumination + 1e-7).pow(
                gamma.view(-1, 1, 1, 1) - 1.0
            ).clamp(0.0, 1.0)
        else:
            gamma = torch.ones((b, 1), device=degraded.device, dtype=degraded.dtype)
            effective_attenuation = base_illumination
        low_light_clean = (clean * effective_attenuation).clamp(0.0, 1.0)

        if not self.haze_decomposition_enabled:
            depth = raster.depth if raster.depth.dim() == 4 else raster.depth.unsqueeze(0)
            transmission = torch.ones_like(depth)
            optical_depth = torch.zeros_like(depth)
            path_radiance = torch.zeros_like(low_light_clean)
            haze_depth = depth
            continuous = low_light_clean
            beta = self.medium.beta
            airlight = self.medium.airlight
        elif self.csm_enabled:
            if camera is None:
                raise ValueError("CSM requires the viewpoint camera")
            csm = synthesize_fogged_image(
                camera, low_light_clean.squeeze(0), raster.depth.squeeze(0).clamp_min(1e-6).reciprocal(),
                self.csm_fog, self.csm_airlight, num_samples=self.csm_num_samples,
            )
            transmission = csm["transmission"].unsqueeze(0)
            path_radiance = csm["C_haze"].permute(2, 0, 1).unsqueeze(0)
            continuous = csm["fogged"].unsqueeze(0)
            optical_depth = -torch.log(transmission.clamp_min(1e-6))
            haze_depth = raster.depth if raster.depth.dim() == 4 else raster.depth.unsqueeze(0)
            beta = self.csm_fog.voxel_grid.voxel_features.mean().view(1)
            airlight = csm["airlight"].mean(dim=(0, 1)).view(-1)
        else:
            transmission, optical_depth, path_radiance, haze_depth = self.medium(raster.depth)
            continuous = (low_light_clean * transmission + path_radiance).clamp(0.0, 1.0)
            beta = self.medium.beta
            airlight = self.medium.airlight

        residual_abs = (degraded - continuous).abs()
        residual_pos = (degraded - continuous).clamp_min(0.0)
        residual_neg = (continuous - degraded).clamp_min(0.0)

        particle_enabled = bool(particle_enabled and self.particle_decomposition_enabled)
        if particle_enabled:
            if particle_residual is None:
                # A particulate layer is a stored training-view buffer.  Never
                # derive it from the target image implicitly during inference.
                particle_residual = torch.zeros_like(continuous)
            else:
                particle_residual = particle_residual.to(
                    device=continuous.device, dtype=continuous.dtype
                ).detach()
                if particle_residual.dim() == 3:
                    particle_residual = particle_residual.unsqueeze(0)
                if particle_residual.shape[-2:] != continuous.shape[-2:]:
                    particle_residual = torch.nn.functional.interpolate(
                        particle_residual,
                        continuous.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
            intensity = particle_residual.mean(dim=1, keepdim=True)
            robust_scale = torch.quantile(
                intensity.flatten(1), 0.95, dim=1
            ).view(-1, 1, 1, 1).detach()
            particle_mask = (intensity / robust_scale.clamp_min(1e-6)).clamp(0.0, 1.0)
            if particle_mask_override is not None:
                particle_mask = particle_mask_override.to(
                    device=continuous.device, dtype=continuous.dtype
                ).detach()
                if particle_mask.dim() == 3:
                    particle_mask = particle_mask.unsqueeze(0)
                if particle_mask.shape[-2:] != continuous.shape[-2:]:
                    particle_mask = torch.nn.functional.interpolate(
                        particle_mask, continuous.shape[-2:], mode="nearest"
                    )
                particle_mask = particle_mask[:, :1].clamp(0.0, 1.0)
                particle_residual = particle_residual * particle_mask
        else:
            particle_residual = torch.zeros_like(continuous)
            particle_mask = torch.zeros_like(continuous[:, :1])
        particle_layer = particle_residual
        strength = min(max(float(particle_strength), 0.0), 1.0)
        final = (continuous + strength * particle_layer).clamp(0.0, 1.0)
        reliability = pixel_reliability(
            effective_attenuation,
            transmission,
            strength * particle_mask,
            low_light_weight=low_light_trust_weight,
            transmission_weight=transmission_trust_weight,
            particle_weight=particle_trust_weight,
        )

        return PhysicsOutput(
            clean_rgb=clean,
            low_light_clean=low_light_clean,
            effective_attenuation=effective_attenuation,
            transmission=transmission,
            optical_depth=optical_depth,
            path_radiance=path_radiance,
            continuous_degraded=continuous,
            residual_abs=residual_abs,
            residual_pos=residual_pos,
            residual_neg=residual_neg,
            particle_mask=particle_mask,
            particle_residual=particle_residual,
            particle_layer=particle_layer,
            final=final,
            pixel_reliability=reliability,
            depth=raster.depth if raster.depth.dim() == 4 else raster.depth.unsqueeze(0),
            beta=beta,
            airlight=airlight,
            base_illumination=base_illumination,
            gamma=gamma,
            haze_depth=haze_depth,
        )
