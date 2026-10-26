from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _to_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"Expected CHW or BCHW tensor, got {tuple(x.shape)}")
    return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(max(1, min(8, out_channels // 4)), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LowLightAttenuationDecoder(nn.Module):
    """Predict E_v^low directly from rendered features and view geometry."""

    def __init__(
        self,
        feature_channels: int,
        view_condition_dim: int = 8,
        hidden: int = 64,
        grid_size: int = 16,
        mode: str = "decoder",
    ):
        super().__init__()
        self.grid_size = max(2, int(grid_size))
        self.mode = str(mode)
        if self.mode not in {"decoder", "view_lowrank"}:
            raise ValueError(f"Unknown illumination mode: {self.mode}")
        in_channels = feature_channels + 1 + 3 + view_condition_dim
        if self.mode == "decoder":
            self.net = nn.Sequential(
                ConvBlock(in_channels, hidden),
                ConvBlock(hidden, hidden),
                nn.Conv2d(hidden, 1, 1),
                nn.Sigmoid(),
            )
            self.view_grid = None
        else:
            # A scene-content-free low-rank illumination field. It cannot read
            # Gaussian features, depth, normals, or pixel-level image content.
            self.net = None
            self.view_grid = nn.Sequential(
                nn.Linear(view_condition_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, self.grid_size * self.grid_size),
                nn.Sigmoid(),
            )

    def forward(
        self,
        illum_features: torch.Tensor,
        depth: torch.Tensor,
        normal: torch.Tensor,
        view_condition: torch.Tensor,
    ) -> torch.Tensor:
        illum_features = _to_bchw(illum_features)
        depth = _to_bchw(depth)
        normal = _to_bchw(normal)
        if depth.shape[1] != 1:
            depth = depth[:, :1]
        depth_norm = depth / depth.detach().flatten(1).amax(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
        b, _, h, w = illum_features.shape
        gh, gw = min(self.grid_size, h), min(self.grid_size, w)
        if self.mode == "view_lowrank":
            attenuation_low = self.view_grid(view_condition).view(
                b, 1, self.grid_size, self.grid_size
            )
            if (gh, gw) != (self.grid_size, self.grid_size):
                attenuation_low = F.adaptive_avg_pool2d(attenuation_low, (gh, gw))
            return F.interpolate(attenuation_low, (h, w), mode="bilinear", align_corners=False)
        features_low = F.adaptive_avg_pool2d(illum_features, (gh, gw))
        depth_low = F.adaptive_avg_pool2d(depth_norm, (gh, gw))
        normal_low = F.adaptive_avg_pool2d(normal, (gh, gw))
        condition = view_condition.view(b, -1, 1, 1).expand(-1, -1, gh, gw)
        attenuation_low = self.net(torch.cat([features_low, depth_low, normal_low, condition], dim=1))
        return F.interpolate(attenuation_low, (h, w), mode="bilinear", align_corners=False)


class HomogeneousMedium(nn.Module):
    def __init__(
        self,
        beta_init: float = 0.08,
        airlight_init: tuple[float, float, float] = (0.8, 0.8, 0.8),
        normalized_depth: bool = False,
        depth_blur_kernel: int = 21,
        depth_max: float = 0.7,
    ):
        super().__init__()
        self.normalized_depth = bool(normalized_depth)
        self.depth_blur_kernel = max(1, int(depth_blur_kernel))
        if self.depth_blur_kernel % 2 == 0:
            self.depth_blur_kernel += 1
        self.depth_max = float(depth_max)
        self.beta_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(beta_init)))))
        air = torch.tensor(airlight_init, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
        self.airlight_logit = nn.Parameter(torch.logit(air))
        # V2 maps metric 3DGS depth to the bounded [0, depth_max] coordinate
        # used by the synthetic formation process.  Scale is positive.
        self.depth_scale_raw = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(max(self.depth_max, 1e-3))))
        )
        self.depth_bias = nn.Parameter(torch.tensor(0.0))

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.beta_raw)

    @property
    def airlight(self) -> torch.Tensor:
        return torch.sigmoid(self.airlight_logit)

    def haze_coordinate(self, depth: torch.Tensor) -> torch.Tensor:
        depth = _to_bchw(depth)
        if depth.shape[1] != 1:
            depth = depth[:, :1]
        positive_depth = depth.clamp_min(1e-6)
        if not self.normalized_depth:
            return positive_depth
        flat = positive_depth.detach().flatten(1)
        near = flat.quantile(0.05, dim=1).view(-1, 1, 1, 1)
        far = flat.quantile(0.95, dim=1).view(-1, 1, 1, 1)
        normalized = ((positive_depth - near) / (far - near).clamp_min(1e-6)).clamp(0.0, 1.0)
        if self.depth_blur_kernel > 1:
            k = self.depth_blur_kernel
            normalized = F.avg_pool2d(normalized, k, stride=1, padding=k // 2)
        scale = F.softplus(self.depth_scale_raw)
        return (scale * normalized + self.depth_bias).clamp(0.0, self.depth_max)

    def forward(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        haze_depth = self.haze_coordinate(depth)
        optical_depth = self.beta.view(1, 1, 1, 1) * haze_depth
        transmission = torch.exp(-optical_depth).clamp(0.0, 1.0)
        path_radiance = self.airlight.view(1, 3, 1, 1) * (1.0 - transmission)
        return transmission, optical_depth, path_radiance, haze_depth
