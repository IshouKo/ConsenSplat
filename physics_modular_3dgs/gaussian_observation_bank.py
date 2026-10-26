from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.sh_utils import SH2RGB


@dataclass
class GaussianCanonicalTargets:
    rgb: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor
    iteration: int
    leave_one_out: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
    brightness_gain: float = 1.0


def calibrate_target_brightness(
    target: torch.Tensor,
    reference_rgb: torch.Tensor,
    confidence: torch.Tensor,
    usable: torch.Tensor,
    gain_limit: float = 0.15,
    min_confidence: float = 0.5,
    min_points: int = 32,
) -> tuple[torch.Tensor, float]:
    """Anchor a Teacher bank to the student's robust global exposure.

    The inverse-degradation Teacher should correct per-Gaussian colour, but a
    global scale error in illumination/transmission must not drag every base
    colour brighter or darker.  A single bounded RGB gain aligns median
    luminance on high-confidence Gaussians while preserving chromaticity and
    local contrast.
    """
    if target.shape != reference_rgb.shape or target.ndim != 2 or target.shape[-1] != 3:
        raise ValueError("target and reference_rgb must both have shape [N,3]")
    target_f = target.float()
    reference_f = reference_rgb.float()
    target_luma = 0.2126 * target_f[:, 0] + 0.7152 * target_f[:, 1] + 0.0722 * target_f[:, 2]
    reference_luma = (
        0.2126 * reference_f[:, 0]
        + 0.7152 * reference_f[:, 1]
        + 0.0722 * reference_f[:, 2]
    )
    mask = (
        usable.bool()
        & (confidence.float() >= float(min_confidence))
        & torch.isfinite(target_f).all(dim=1)
        & torch.isfinite(reference_f).all(dim=1)
        & (target_luma > 0.02)
        & (reference_luma > 0.02)
    )
    if int(mask.sum()) < int(min_points):
        return target, 1.0
    gain = reference_luma[mask].median() / target_luma[mask].median().clamp_min(1e-4)
    limit = max(0.0, float(gain_limit))
    gain = gain.clamp(1.0 - limit, 1.0 + limit)
    calibrated = torch.where(
        usable[:, None],
        (target_f * gain).clamp(0.0, 1.0),
        target_f,
    )
    return calibrated.to(target.dtype), float(gain)


def robust_luminance_quantile(
    observations: torch.Tensor,
    valid: torch.Tensor,
    quantile: float = 0.8,
    min_views: int = 5,
    mad_scale: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one observed RGB triplet after per-Gaussian MAD rejection.

    observations is [N,V,3] and valid is [N,V].  Ranking complete RGB
    observations by luminance avoids the colour shifts caused by independently
    taking R/G/B quantiles.
    """
    if observations.ndim != 3 or observations.shape[-1] != 3:
        raise ValueError("observations must have shape [N,V,3]")
    if valid.shape != observations.shape[:2]:
        raise ValueError("valid must have shape [N,V]")
    values = observations.float()
    mask = valid.bool() & torch.isfinite(values).all(dim=-1)
    nan_values = values.masked_fill(~mask[..., None], float("nan"))
    median = torch.nanmedian(nan_values, dim=1).values
    distance = (values - median[:, None]).abs().mean(dim=-1)
    distance_nan = distance.masked_fill(~mask, float("nan"))
    mad = torch.nanmedian(distance_nan, dim=1).values
    threshold = float(mad_scale) * mad.clamp_min(0.01)
    inlier = mask & (distance <= threshold[:, None])
    count = inlier.sum(dim=1)

    luminance = 0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]
    luminance = luminance.masked_fill(~inlier, float("inf"))
    order = luminance.argsort(dim=1)
    rank = ((count.clamp_min(1) - 1).float() * float(quantile)).round().long()
    selected_index = order.gather(1, rank[:, None]).squeeze(1)
    target = values.gather(1, selected_index[:, None, None].expand(-1, 1, 3)).squeeze(1)

    selected_luma = luminance.gather(1, selected_index[:, None]).squeeze(1)
    spread = torch.nan_to_num(mad, nan=1.0, posinf=1.0)
    confidence = (count.float() / max(int(min_views), 1)).clamp(max=1.0) * torch.exp(-spread / 0.10)
    usable = (count >= int(min_views)) & torch.isfinite(selected_luma)
    target = torch.where(usable[:, None], target.clamp(0, 1), torch.zeros_like(target))
    confidence = torch.where(usable, confidence, torch.zeros_like(confidence))
    return target, confidence, usable


def persistent_color_mode(
    observations: torch.Tensor,
    valid: torch.Tensor,
    min_views: int = 5,
    mad_scale: float = 2.5,
    num_bins: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate the repeatedly observed surface colour, rejecting transients.

    A dominant luminance bin supplies a cheap mode estimate without allocating
    an O(V^2) pairwise-distance tensor.  A second RGB MAD pass removes rain,
    snow, and reprojection outliers which happen to share that luminance bin.
    """
    if observations.ndim != 3 or observations.shape[-1] != 3:
        raise ValueError("observations must have shape [N,V,3]")
    values = observations.float()
    mask = valid.bool() & torch.isfinite(values).all(dim=-1)
    luma = 0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]
    bins = (luma.clamp(0.0, 1.0) * (int(num_bins) - 1)).round().long()
    counts = values.new_zeros((values.shape[0], int(num_bins)))
    counts.scatter_add_(1, bins, mask.to(values.dtype))
    dominant = counts.argmax(dim=1)
    cluster = mask & (bins == dominant[:, None])

    count0 = cluster.sum(dim=1).clamp_min(1)
    centre0 = (values * cluster[..., None]).sum(dim=1) / count0[:, None]
    distance = (values - centre0[:, None]).abs().mean(dim=-1)
    distance_nan = distance.masked_fill(~cluster, float("nan"))
    mad = torch.nanmedian(distance_nan, dim=1).values
    inlier = cluster & (distance <= (float(mad_scale) * mad.clamp_min(0.015))[:, None])
    count = inlier.sum(dim=1)
    target = (values * inlier[..., None]).sum(dim=1) / count.clamp_min(1)[:, None]
    valid_count = mask.sum(dim=1).clamp_min(1)
    support = count.float() / valid_count.float()
    spread = torch.nan_to_num(mad, nan=1.0, posinf=1.0)
    confidence = support * torch.exp(-spread / 0.08)
    usable = count >= int(min_views)
    target = torch.where(usable[:, None], target.clamp(0, 1), torch.zeros_like(target))
    confidence = torch.where(usable, confidence, torch.zeros_like(confidence))
    return target, confidence, usable


def project_gaussians(camera, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ones = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
    homogeneous = torch.cat((xyz, ones), dim=1)
    clip = homogeneous @ camera.full_proj_transform.to(xyz)
    safe_w = torch.where(clip[:, 3].abs() > 1e-6, clip[:, 3], torch.ones_like(clip[:, 3]))
    grid = clip[:, :2] / safe_w[:, None]
    camera_xyz = homogeneous @ camera.world_view_transform.to(xyz)
    depth = camera_xyz[:, 2] / torch.where(
        camera_xyz[:, 3].abs() > 1e-6, camera_xyz[:, 3], torch.ones_like(camera_xyz[:, 3])
    )
    inside = (clip[:, 3] > 0) & (depth > 0) & (grid.abs() <= 1).all(dim=1)
    return grid, depth, inside


def sample_map(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    image = image if image.ndim == 4 else image.unsqueeze(0)
    return F.grid_sample(
        image, grid.view(1, -1, 1, 2), mode="bilinear",
        padding_mode="zeros", align_corners=True,
    ).squeeze(0).squeeze(-1).transpose(0, 1)


def residual_base_targets(
    corrected_rgb: torch.Tensor,
    rendered_rgb: torch.Tensor,
    base_rgb: torch.Tensor,
    max_delta: float,
) -> torch.Tensor:
    """Transfer inverse-image residuals to base colours with a trust region.

    Rendered pixels are alpha-composited mixtures, so treating their absolute
    RGB as the colour of each projected Gaussian is biased. Their difference
    from the current render is nevertheless a useful local correction signal.
    """
    limit = max(0.0, float(max_delta))
    residual = corrected_rgb - rendered_rgb
    if limit > 0.0:
        residual = residual.clamp(min=-limit, max=limit)
    return (base_rgb + residual).clamp(0.0, 1.0)


def gauge_fix_rgb_update(
    target_rgb: torch.Tensor,
    reference_rgb: torch.Tensor,
    confidence: torch.Tensor,
    usable: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove the confidence-weighted global RGB shift from a Teacher target.

    The physics decomposition cannot identify a simultaneous global change in
    canonical colour and inverse change in illumination. Keeping the Teacher
    correction zero-mean leaves that gauge with the DCP/base reconstruction
    anchor while preserving local and structural colour corrections.
    """
    if target_rgb.shape != reference_rgb.shape or target_rgb.shape[-1] != 3:
        raise ValueError("target_rgb and reference_rgb must have shape [N,3]")
    weight = confidence.float() * usable.float()
    denominator = weight.sum().clamp_min(1e-6)
    offset = ((target_rgb.float() - reference_rgb.float()) * weight[:, None]).sum(0) / denominator
    fixed = (target_rgb.float() - offset).clamp(0.0, 1.0)
    fixed = torch.where(usable[:, None], fixed, target_rgb.float())
    return fixed.to(target_rgb.dtype), offset


class GaussianObservationBank:
    """EMA Gaussian teacher with leave-one-observation-out targets.

    Views which were not sampled by a refresh are already leave-one-out by
    construction.  For every sampled view we additionally retain a target
    rebuilt with that observation removed, so a camera can never supervise
    itself through the teacher.
    """

    def __init__(self, quantile=0.8, min_views=5, mad_scale=2.5, ema_momentum=0.9,
                 estimator="quantile", brightness_calibration=False,
                 brightness_gain_limit=0.15, brightness_min_confidence=0.5,
                 loss_beta=1.0):
        self.quantile = float(quantile)
        self.min_views = int(min_views)
        self.mad_scale = float(mad_scale)
        self.ema_momentum = float(ema_momentum)
        self.estimator = str(estimator)
        self.brightness_calibration = bool(brightness_calibration)
        self.brightness_gain_limit = float(brightness_gain_limit)
        self.brightness_min_confidence = float(brightness_min_confidence)
        self.loss_beta = float(loss_beta)
        self.targets: GaussianCanonicalTargets | None = None

    def _ema(self, old, new):
        if old is None or old.shape != new.shape:
            return new.detach()
        return (self.ema_momentum * old.to(new) + (1.0 - self.ema_momentum) * new).detach()

    def build(
        self, observations: torch.Tensor, valid: torch.Tensor, iteration: int,
        view_ids=None, use_ema: bool = True, reference_rgb: torch.Tensor | None = None,
    ):
        estimate = persistent_color_mode if self.estimator == "persistent_mode" else robust_luminance_quantile
        kwargs = {"min_views": self.min_views, "mad_scale": self.mad_scale}
        if estimate is robust_luminance_quantile:
            kwargs["quantile"] = self.quantile
        rgb, confidence, usable = estimate(observations, valid, **kwargs)
        brightness_gain = 1.0
        if self.brightness_calibration:
            if reference_rgb is None:
                raise ValueError("brightness calibration requires reference_rgb")
            rgb, brightness_gain = calibrate_target_brightness(
                rgb, reference_rgb, confidence, usable,
                self.brightness_gain_limit, self.brightness_min_confidence,
            )
        # Before densification ends, Gaussian identities are not persistent.
        # Rebuilding without EMA prevents a child's target from being blended
        # with an unrelated point that happened to reuse the same row index.
        old = self.targets if use_ema else None
        rgb = self._ema(old.rgb if old is not None else None, rgb)
        confidence = self._ema(old.confidence if old is not None else None, confidence)
        leave_one_out = {}
        if view_ids is not None:
            for slot, view_id in enumerate(view_ids):
                loo_valid = valid.clone()
                loo_valid[:, slot] = False
                loo_rgb, loo_confidence, loo_usable = estimate(observations, loo_valid, **kwargs)
                if self.brightness_calibration:
                    loo_rgb, _ = calibrate_target_brightness(
                        loo_rgb, reference_rgb, loo_confidence, loo_usable,
                        self.brightness_gain_limit, self.brightness_min_confidence,
                    )
                old_loo = old.leave_one_out.get(int(view_id)) if old is not None and old.leave_one_out else None
                loo_rgb = self._ema(old_loo[0] if old_loo else None, loo_rgb)
                loo_confidence = self._ema(old_loo[1] if old_loo else None, loo_confidence)
                leave_one_out[int(view_id)] = (
                    loo_rgb.to(torch.float16), loo_confidence.to(torch.float16), loo_usable.detach()
                )
        self.targets = GaussianCanonicalTargets(
            rgb.to(torch.float16), confidence.to(torch.float16), usable.detach(), int(iteration),
            leave_one_out, brightness_gain,
        )
        return self.targets

    def loss(
        self,
        features_dc: torch.Tensor,
        view_id: int | None = None,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
    ) -> torch.Tensor:
        if self.targets is None:
            return features_dc.sum() * 0.0
        current = SH2RGB(features_dc[:, 0, :])
        if current.shape[0] != self.targets.rgb.shape[0]:
            raise RuntimeError("Gaussian count changed after the observation bank was built")
        selected = self.targets.leave_one_out.get(int(view_id)) if view_id is not None and self.targets.leave_one_out else None
        target, confidence, usable = selected if selected is not None else (
            self.targets.rgb, self.targets.confidence, self.targets.valid
        )
        per_point = F.smooth_l1_loss(
            current, target.to(current), reduction="none",
            beta=max(float(self.loss_beta), 1e-6),
        ).mean(dim=1)
        confidence_current = confidence.to(current)
        confidence_band = (
            (confidence_current >= float(min_confidence))
            & (confidence_current <= float(max_confidence))
        )
        weight = confidence_current * usable.to(current) * confidence_band.to(current)
        return (per_point * weight).sum() / weight.sum().clamp_min(1e-6)

    def selected(self, view_id: int | None = None):
        """Return the frozen global or leave-one-view-out target tuple."""
        if self.targets is None:
            return None
        selected = (
            self.targets.leave_one_out.get(int(view_id))
            if view_id is not None and self.targets.leave_one_out else None
        )
        return selected if selected is not None else (
            self.targets.rgb, self.targets.confidence, self.targets.valid
        )
