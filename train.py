from __future__ import annotations

import json
import os
import random
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, parse_configured_args
from gaussian_renderer import render
from physics_modular_3dgs.integration import (
    load_optional_reference,
    make_training_batch,
    project_reliability_to_gaussians,
    run_physics,
    save_physics_state,
    write_run_metadata,
)
from physics_modular_3dgs.losses import compute_physics_losses
from physics_modular_3dgs.model import PhysicsAware3DGS
from physics_modular_3dgs.multiview_consensus import (
    ConsensusCache, invert_degradation, nearest_camera_map, reproject_consensus,
)
from physics_modular_3dgs.gaussian_observation_bank import (
    GaussianObservationBank, project_gaussians, sample_map,
)
from physics_modular_3dgs.particle_supervision import ParticlePseudoLabelCache
from physics_modular_3dgs.training_schedule import physics_prediction_weight, particle_strength, training_stage, trust_weights
from physics_modular_3dgs.types import LossWeights
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import LPIPS
from dark_channel_prior import dark_channel_prior

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False

_LPIPS_VGG = None


def build_descriptor(mode, _checkpoint, _device):
    if str(mode).lower() not in {"", "none", "off", "disabled"}:
        raise RuntimeError("Degradation-descriptor branches are not included in this focused copy")
    return None


def descriptor_losses(*_args, **_kwargs):
    raise RuntimeError("Degradation-descriptor branches are not included in this focused copy")


def build_onerestore_teacher(enabled, *_args, **_kwargs):
    if enabled:
        raise RuntimeError("OneRestore branches are not included in this degraded-only focused copy")
    return None


def crossview_objective(*_args, **_kwargs):
    raise RuntimeError("Cross-view VarPro branches are not included in this focused copy")


def _box_blur(image: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    kernel_size = max(3, int(kernel_size) | 1)
    return F.avg_pool2d(image, kernel_size, stride=1, padding=kernel_size // 2)


def _edge_strength(image: torch.Tensor) -> torch.Tensor:
    gray = 0.2126 * image[:, :1] + 0.7152 * image[:, 1:2] + 0.0722 * image[:, 2:3]
    dx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
    magnitude = (dx.square() + dy.square() + 1e-8).sqrt()
    scale = torch.quantile(magnitude.flatten(1), 0.95, dim=1).view(-1, 1, 1, 1)
    return (magnitude / scale.clamp_min(1e-4)).clamp(0.0, 1.0)


def _persistent_and_transient_maps(consensus_label, output, observed):
    confidence = consensus_label.confidence.to(output.clean_rgb).clamp(0.0, 1.0)
    persistent = _edge_strength(consensus_label.clean.to(output.clean_rgb)) * confidence
    observed = observed if observed.dim() == 4 else observed.unsqueeze(0)
    residual = (observed.to(output.continuous_degraded) - output.continuous_degraded).abs().mean(1, keepdim=True)
    high = (residual - _box_blur(residual)).abs()
    scale = torch.quantile(high.flatten(1), 0.95, dim=1).view(-1, 1, 1, 1)
    transient = (high / scale.clamp_min(1e-4)).clamp(0.0, 1.0)
    transient = transient * (1.0 - confidence) * (1.0 - persistent)
    return persistent.detach(), transient.detach()


def _frequency_consensus_loss(prediction, target, confidence):
    target = target.to(prediction).detach()
    confidence = confidence.to(prediction).detach().clamp(0.0, 1.0)
    pred_low, target_low = _box_blur(prediction), _box_blur(target)
    pred_high, target_high = prediction - pred_low, target - target_low
    weight = confidence.expand_as(prediction)
    denom = weight.sum().clamp_min(1.0)
    low = ((pred_low - target_low).abs() * weight).sum() / denom
    high = ((pred_high - target_high).abs() * weight).sum() / denom
    return low + high

def _lpips_vgg(device):
    global _LPIPS_VGG
    if _LPIPS_VGG is None:
        _LPIPS_VGG = LPIPS("vgg").to(device).eval()
        for parameter in _LPIPS_VGG.parameters():
            parameter.requires_grad_(False)
    return _LPIPS_VGG


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Full 3DGS training requires an NVIDIA CUDA environment. "
            "The branch-only smoke test can still run with: "
            "python -m physics_modular_3dgs.train --smoke-test"
        )


def _loss_weights(opt) -> LossWeights:
    return LossWeights(
        image=opt.lambda_image,
        low_light_attenuation=opt.lambda_low_light_attenuation,
        transmission=opt.lambda_transmission,
        tau=opt.lambda_tau_physics,
        path_radiance=opt.lambda_path_radiance,
        clean_pseudo=(opt.lambda_consensus_clean if opt.degraded_only else opt.lambda_clean_pseudo),
        illumination_anchor=(opt.lambda_illumination_anchor if opt.degraded_only else 0.0),
        consensus_ssim=(opt.lambda_consensus_ssim if opt.degraded_only else 0.0),
        consensus_perceptual=(opt.lambda_consensus_perceptual if opt.degraded_only else 0.0),
        consensus_gradient=(opt.lambda_consensus_gradient if opt.degraded_only else 0.0),
        beta=opt.lambda_beta,
        airlight=opt.lambda_airlight,
        depth_tv=opt.lambda_depth_tv,
        low_light_smooth=opt.lambda_low_light_smooth,
        ssim=opt.lambda_dssim,
    )


def _apply_stage_geometry_lrs(gaussians, opt, stage: str) -> None:
    # update_learning_rate() resets xyz first; assign all other rates from their
    # immutable bases so stage multipliers never compound across iterations.
    geometry_multiplier = opt.stage2_geometry_lr_multiplier if stage == "particle_init" else 1.0
    scale_multiplier = opt.stage1_scale_lr_multiplier if stage == "geometry_medium_init" else geometry_multiplier
    feature_multiplier = opt.stage1_feature_lr_multiplier if stage == "geometry_medium_init" else geometry_multiplier
    for group in gaussians.optimizer.param_groups:
        name = group.get("name")
        if name == "xyz":
            group["lr"] = group["lr"] * geometry_multiplier
        elif name == "scaling":
            group["lr"] = opt.scaling_lr * scale_multiplier
        elif name == "rotation":
            group["lr"] = opt.rotation_lr * geometry_multiplier
        elif name == "f_dc":
            group["lr"] = opt.feature_lr * feature_multiplier
        elif name == "f_rest":
            group["lr"] = opt.feature_lr / 20.0 * feature_multiplier


def _apply_appearance_only_lrs(gaussians) -> None:
    """Freeze geometry and per-Gaussian physics features, retaining SH colour."""
    appearance_groups = {"f_dc", "f_rest"}
    for group in gaussians.optimizer.param_groups:
        if group.get("name") not in appearance_groups:
            group["lr"] = 0.0


def _apply_physics_feature_only_lrs(gaussians) -> None:
    """Freeze canonical 3DGS parameters while retaining learned physics features."""
    physics_feature_groups = {"f_illum", "f_degradation"}
    for group in gaussians.optimizer.param_groups:
        if group.get("name") not in physics_feature_groups:
            group["lr"] = 0.0


def _apply_fixed_radiance_lrs(gaussians) -> None:
    """Keep DCP-initialized SH radiance fixed while geometry may densify."""
    for group in gaussians.optimizer.param_groups:
        if group.get("name") in {"f_dc", "f_rest"}:
            group["lr"] = 0.0


def _apply_dcp_residual_teacher_lrs(gaussians, opt, iteration: int, physics_frozen: bool) -> None:
    """Keep an imported DCP scene fixed except for a bounded SH-DC residual.

    Illumination/degradation embeddings remain trainable while the physical
    model is active.  The DC colour is held exactly at its DCP initialization
    until the Gaussian teacher has enough observations to become active.
    """
    allowed = {"f_dc"}
    if not physics_frozen:
        allowed.update({"f_illum", "f_degradation"})
    for group in gaussians.optimizer.param_groups:
        name = group.get("name")
        if name not in allowed:
            group["lr"] = 0.0
        elif name == "f_dc":
            teacher_window_active = (
                iteration >= int(opt.dcp_residual_teacher_start_iter)
                and (
                    int(opt.gaussian_consensus_end) <= 0
                    or iteration <= int(opt.gaussian_consensus_end)
                )
            )
            group["lr"] = (
                float(opt.feature_lr)
                if teacher_window_active
                else 0.0
            )


def _prepare_output(dataset, args: Namespace) -> None:
    if not dataset.model_path:
        dataset.model_path = os.path.join("output", "physics_3dgs")
        args.model_path = dataset.model_path
    Path(dataset.model_path).mkdir(parents=True, exist_ok=True)
    with (Path(dataset.model_path) / "cfg_args").open("w", encoding="utf-8") as handle:
        handle.write(str(Namespace(**vars(args))))


def _validate_degraded_analytic_prior(dataset, opt) -> None:
    """Require proof that an analytic clean prior was derived from active inputs."""
    if (
        float(opt.lambda_degraded_analytic_prior) <= 0.0
        and not bool(opt.dcp_residual_teacher_enabled)
    ):
        return
    if not opt.degraded_only:
        raise ValueError("degraded_analytic_prior requires degraded_only training")
    prior_dir = Path(opt.degraded_analytic_prior_dir).expanduser().resolve()
    metadata_path = prior_dir.parent / "degraded_frontend.json"
    if not prior_dir.is_dir() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Analytic prior or provenance metadata missing: {prior_dir}, {metadata_path}"
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    active_images = (Path(dataset.source_path).expanduser().resolve() / dataset.images).resolve()
    source_images = Path(metadata.get("degraded_input", "")).expanduser().resolve()
    if metadata.get("training_inputs") != ["degraded_input"] or source_images != active_images:
        raise ValueError(
            "Analytic prior provenance does not match the active degraded-only image directory"
        )
    print(f"Verified degraded-only analytic prior provenance: {metadata_path}", flush=True)


@torch.no_grad()
def _validate(scene, gaussians, physics, pipe, background, feature_channels, clean_dir, iteration) -> dict[str, float]:
    cameras = scene.getTestCameras() or scene.getTrainCameras()[: min(8, len(scene.getTrainCameras()))]
    if not cameras:
        return {}
    was_training = physics.training
    physics.eval()
    criterion = _lpips_vgg(background.device)
    per_image = {}
    for camera in cameras:
        package = render(
            camera, gaussians, pipe, background,
            separate_sh=SPARSE_ADAM_AVAILABLE, render_features=True)
        view_index = min(int(camera.uid), physics.view_condition.num_embeddings - 1)
        output, _ = run_physics(physics, package, camera.original_image, view_index, feature_channels, camera=camera)
        predicted = output.clean_rgb.clamp(0, 1)
        target = load_optional_reference(
            clean_dir, camera.image_name, predicted.shape[-2:], predicted.device
        )
        if target is None:
            continue
        target = target.clamp(0, 1)
        degraded_target = camera.original_image.to(output.final).unsqueeze(0).clamp(0, 1)
        degraded_prediction = output.final.clamp(0, 1)
        per_image[camera.image_name] = {
            "PSNR": float(psnr(predicted, target).mean()),
            "SSIM": float(ssim(predicted.squeeze(0), target.squeeze(0)).mean()),
            "LPIPS": float(criterion(predicted, target).mean()),
            "DegradedPSNR": float(psnr(degraded_prediction, degraded_target).mean()),
            "DegradedSSIM": float(
                ssim(degraded_prediction.squeeze(0), degraded_target.squeeze(0)).mean()
            ),
        }
    physics.train(was_training)
    if not per_image:
        return {}
    metrics = {
        name: sum(item[name] for item in per_image.values()) / len(per_image)
        for name in ("PSNR", "SSIM", "LPIPS", "DegradedPSNR", "DegradedSSIM")
    }
    metrics_dir = Path(scene.model_path) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / f"test_iteration_{iteration}.json").open("w", encoding="utf-8") as handle:
        json.dump({"average": metrics, "per_image": per_image, "num_images": len(per_image)}, handle, indent=2)
    return metrics


@torch.no_grad()
def _get_consensus_label(cache, neighbors_by_name, camera, raster, iteration, start_iteration, gaussians, physics, pipe, background, feature_channels, observed_dir=""):
    if iteration < int(start_iteration):
        return None
    cached = cache.get(camera.image_name, raster.clean_rgb.device, raster.clean_rgb.dtype)
    if cached is not None and not cache.needs_refresh(camera.image_name, iteration):
        return cached
    sources = []
    for source in neighbors_by_name.get(camera.image_name, []):
        source_package = render(
            source, gaussians, pipe, background,
            separate_sh=SPARSE_ADAM_AVAILABLE, render_features=True,
        )
        source_observed = load_optional_reference(
            observed_dir, source.image_name, source.original_image.shape[-2:], raster.clean_rgb.device
        ) if observed_dir else source.original_image.to(raster.clean_rgb.device)
        if source_observed is None:
            continue
        source_index = min(int(source.uid), physics.view_condition.num_embeddings - 1)
        source_output, source_raster = run_physics(
            physics, source_package, source_observed, source_index, feature_channels,
            particle_enabled=False,
            camera=source,
        )
        source_clean = invert_degradation(
            source_observed, source_output.effective_attenuation,
            source_output.transmission, source_output.path_radiance,
        )
        sources.append((source, source_clean, source_raster.depth.detach()))
    if not sources:
        return None
    clean, confidence = reproject_consensus(
        camera, raster.depth.detach(), sources,
        agreement_scale=cache.agreement_scale,
        depth_consistency=cache.depth_consistency,
        min_views=cache.min_views,
        confidence_threshold=cache.confidence_threshold,
    )
    return cache.put(camera.image_name, clean, confidence, iteration)


def _checkpoint_payload(iteration, gaussians, physics, physics_optimizer, feature_channels, num_views, stage, particle_pseudo):
    return {
        "iteration": iteration,
        "gaussians": gaussians.capture(),
        "physics": physics.state_dict(),
        "physics_optimizer": physics_optimizer.state_dict(),
        "feature_channels": feature_channels,
        "num_views": num_views,
        "stage": stage,
        "particle_buffers": particle_pseudo.state_dict(),
    }


@torch.no_grad()
def _refresh_gaussian_observation_bank(
    bank, cameras, gaussians, physics, pipe, background, feature_channels, opt, iteration,
):
    """Build frozen canonical-colour targets from real degraded observations."""
    max_views = (
        int(opt.gaussian_consensus_max_views_pre_densify)
        if iteration < int(opt.gaussian_consensus_ema_start)
        else int(opt.gaussian_consensus_max_views)
    )
    if len(cameras) > max_views:
        indices = torch.linspace(
            0, len(cameras) - 1, max_views
        ).round().long().tolist()
        cameras = [cameras[index] for index in indices]
    num_points = gaussians.get_xyz.shape[0]
    observations = torch.zeros(
        (num_points, len(cameras), 3), device=gaussians.get_xyz.device, dtype=torch.float16
    )
    valid = torch.zeros(
        (num_points, len(cameras)), device=gaussians.get_xyz.device, dtype=torch.bool
    )
    was_training = physics.training
    physics.eval()
    for view_slot, source in enumerate(cameras):
        package = render(
            source, gaussians, pipe, background,
            separate_sh=SPARSE_ADAM_AVAILABLE, render_features=True,
        )
        observed = source.original_image.to(device=background.device)
        view_index = min(int(source.uid), physics.view_condition.num_embeddings - 1)
        output, raster = run_physics(
            physics, package, observed, view_index, feature_channels, particle_enabled=True,
            camera=source,
        )
        corrected = invert_degradation(
            observed, output.effective_attenuation,
            output.transmission, output.path_radiance,
        )
        grid, gaussian_depth, inside = project_gaussians(source, gaussians.get_xyz)
        sampled_rgb = sample_map(corrected, grid)
        sampled_depth = sample_map(raster.depth, grid).squeeze(1)
        sampled_transmission = sample_map(output.transmission, grid).squeeze(1)
        sampled_reliability = sample_map(output.pixel_reliability, grid).squeeze(1)
        sampled_particle = sample_map(output.particle_mask, grid).squeeze(1)
        depth_error = (sampled_depth - gaussian_depth).abs() / sampled_depth.abs().clamp_min(1e-3)
        view_valid = (
            inside
            & package["visibility_filter"]
            & (package["radii"] >= float(opt.gaussian_consensus_min_radius))
            & (sampled_depth > 0)
            & (depth_error <= float(opt.gaussian_consensus_depth_consistency))
            & (sampled_transmission >= float(opt.gaussian_consensus_min_transmission))
            & (sampled_reliability >= float(opt.gaussian_consensus_min_reliability))
            & (sampled_particle < 0.5)
            & torch.isfinite(sampled_rgb).all(dim=1)
        )
        observations[:, view_slot] = sampled_rgb.to(torch.float16)
        valid[:, view_slot] = view_valid
        del package, output, raster, corrected, sampled_rgb
    targets = bank.build(
        observations, valid, iteration, [int(camera.uid) for camera in cameras],
        use_ema=iteration >= int(opt.gaussian_consensus_ema_start),
    )
    physics.train(was_training)
    coverage = float(targets.valid.float().mean())
    mean_views = float(valid.sum(dim=1)[targets.valid].float().mean()) if targets.valid.any() else 0.0
    print(
        f"\n[ITER {iteration}] V1-GQ bank: views={len(cameras)}, "
        f"valid_gaussians={int(targets.valid.sum())}/{num_points} "
        f"({coverage:.1%}), mean_valid_views={mean_views:.2f}"
    )


def training(dataset, opt, pipe, args) -> None:
    _require_cuda()
    _validate_degraded_analytic_prior(dataset, opt)
    if opt.dcp_residual_teacher_enabled:
        if not opt.degraded_only:
            raise ValueError("DCP residual teacher requires degraded_only training")
        if not opt.gaussian_init_ply:
            raise ValueError("DCP residual teacher requires gaussian_init_ply")
        if not opt.gaussian_consensus_enabled:
            raise ValueError("DCP residual teacher requires gaussian_consensus_enabled")
        if opt.freeze_canonical_gaussians or opt.freeze_canonical_radiance:
            raise ValueError("DCP residual teacher is incompatible with canonical freeze modes")
        if float(opt.dcp_residual_max_rgb_delta) <= 0.0:
            raise ValueError("dcp_residual_max_rgb_delta must be positive")
    if opt.crossview_enabled:
        if not opt.degraded_only or not opt.fixed_gaussian_geometry:
            raise ValueError("Cross-view pilot requires degraded_only and fixed_gaussian_geometry")
        if not args.start_checkpoint:
            raise ValueError("Cross-view pilot requires a trained geometry checkpoint")
        if opt.gaussian_consensus_enabled or opt.onerestore_teacher_enabled:
            raise ValueError("Cross-view pilot replaces the Gaussian/OneRestore teachers")
        if opt.gaussian_appearance_only_start_iter > 0 or opt.physics_freeze_iter > 0:
            raise ValueError("Cross-view pilot must retain physics optimizer steps")
        if opt.crossview_interval < 1 or opt.crossview_neighbors < opt.crossview_min_views:
            raise ValueError("Invalid cross-view schedule or support-view count")
        if min(opt.alternating_gaussian_steps, opt.alternating_physics_steps) < 1:
            raise ValueError("Cross-view pilot needs both Gaussian and physics phases")
        if opt.random_background or opt.csm_enabled or opt.lambda_clean_oracle > 0:
            raise ValueError("Pilot requires fixed background, homogeneous medium and no clean oracle")
    _prepare_output(dataset, args)
    gaussians = GaussianModel(
        dataset.sh_degree, opt.optimizer_type, feature_channels=opt.physics_feature_channels)
    scene = Scene(dataset, gaussians)
    if opt.gaussian_init_ply:
        init_ply = Path(opt.gaussian_init_ply).expanduser().resolve()
        if not init_ply.is_file():
            raise FileNotFoundError(f"Gaussian initialization PLY not found: {init_ply}")
        gaussians.load_ply(str(init_ply))
        print(
            f"Initialized {gaussians.get_xyz.shape[0]} Gaussians from {init_ply}",
            flush=True,
        )
    gaussians.training_setup(opt)

    train_cameras = scene.getTrainCameras()
    all_cameras = train_cameras + scene.getTestCameras()
    num_views = max((int(camera.uid) for camera in all_cameras), default=0) + 1
    def make_physics():
        return PhysicsAware3DGS(
            illum_feature_channels=opt.physics_feature_channels,
            degradation_feature_channels=opt.physics_feature_channels,
            num_views=num_views,
            beta_init=opt.beta_init,
            illumination_grid_size=opt.illumination_grid_size,
            illumination_mode=opt.illumination_mode,
            physics_v2=opt.physics_v2,
            gamma_min=opt.gamma_min,
            gamma_max=opt.gamma_max,
            haze_depth_normalized=opt.haze_depth_normalized,
            haze_depth_blur_kernel=opt.haze_depth_blur_kernel,
            haze_depth_max=opt.haze_depth_max,
            csm_enabled=opt.csm_enabled,
            csm_gaussians=gaussians,
            csm_resolution=opt.csm_resolution,
            csm_num_samples=opt.csm_num_samples,
            csm_density_scale_init=opt.csm_density_scale_init,
            csm_airlight_mode=opt.csm_airlight_mode,
            haze_decomposition_enabled=opt.haze_decomposition_enabled,
            particle_decomposition_enabled=opt.particle_decomposition_enabled,
        )

    condition_roots = [Path(item) for item in str(opt.multi_condition_dirs).split(";") if item.strip()]
    condition_image_dirs = [root / "images" for root in condition_roots]
    for image_dir in condition_image_dirs:
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Multi-condition image directory not found: {image_dir}")
    physics_models = nn.ModuleList([make_physics() for _ in range(max(1, len(condition_roots)))]).cuda()
    physics = physics_models[0]
    physics_optimizer = torch.optim.AdamW(
        physics_models.parameters(), lr=opt.physics_lr, weight_decay=opt.physics_weight_decay
    )
    if opt.physics_init_path:
        physics_init_path = Path(opt.physics_init_path).expanduser().resolve()
        if not physics_init_path.is_file():
            raise FileNotFoundError(f"Physics initialization state not found: {physics_init_path}")
        physics_state = torch.load(physics_init_path, map_location="cuda", weights_only=False)
        if len(physics_models) != 1:
            raise ValueError("physics_init_path currently supports a single condition")
        physics_models[0].load_state_dict(physics_state["model"], strict=True)
        print(f"Initialized physics model from {physics_init_path}", flush=True)
    first_iteration = 0
    restored_particle_buffers = None

    if args.start_checkpoint:
        checkpoint = torch.load(args.start_checkpoint, map_location="cuda", weights_only=False)
        if isinstance(checkpoint, dict):
            gaussians.restore(checkpoint["gaussians"], opt)
            state = checkpoint["physics"]
            if len(physics_models) == 1 and not any(str(key).startswith("0.") for key in state):
                physics_models[0].load_state_dict(state, strict=False)
            else:
                physics_models.load_state_dict(state, strict=False)
            if "physics_optimizer" in checkpoint:
                try:
                    physics_optimizer.load_state_dict(checkpoint["physics_optimizer"])
                except ValueError:
                    print("Ignoring an incompatible legacy physics optimizer state.")
            restored_particle_buffers = checkpoint.get("particle_buffers")
            first_iteration = int(checkpoint.get("iteration", 0))
        else:
            gaussian_state, first_iteration = checkpoint
            gaussians.restore(gaussian_state, opt)

    dcp_anchor_dc = None
    if opt.dcp_residual_teacher_enabled:
        anchor_path = Path(dataset.model_path) / "dcp_anchor_dc.pt"
        if args.start_checkpoint and anchor_path.is_file():
            dcp_anchor_dc = torch.load(anchor_path, map_location="cuda", weights_only=True).to(
                gaussians.get_features_dc
            )
        else:
            dcp_anchor_dc = gaussians.get_features_dc.detach().clone()
            torch.save(dcp_anchor_dc.cpu(), anchor_path)
        if dcp_anchor_dc.shape != gaussians.get_features_dc.shape:
            raise ValueError("Saved DCP anchor does not match the current Gaussian topology")
        allowed = {"f_dc", "f_illum", "f_degradation"}
        for group in gaussians.optimizer.param_groups:
            if group.get("name") not in allowed:
                for parameter in group["params"]:
                    parameter.requires_grad_(False)
        print(
            "DCP residual teacher mode: fixed geometry/opacity/high-order SH; "
            f"SH-DC residual limited to +/-{float(opt.dcp_residual_max_rgb_delta):.3f} RGB.",
            flush=True,
        )
    elif opt.freeze_canonical_gaussians:
        # Preserve the initialized clean representation exactly.  Only the
        # per-Gaussian physics embeddings and the physics network can update.
        _apply_physics_feature_only_lrs(gaussians)
        for group in gaussians.optimizer.param_groups:
            if group.get("name") not in {"f_illum", "f_degradation"}:
                for parameter in group["params"]:
                    parameter.requires_grad_(False)
        print(
            "Frozen canonical Gaussian geometry/radiance; physics features and "
            "physics network remain trainable.",
            flush=True,
        )
    elif opt.freeze_canonical_radiance:
        _apply_fixed_radiance_lrs(gaussians)
        for group in gaussians.optimizer.param_groups:
            if group.get("name") in {"f_dc", "f_rest"}:
                for parameter in group["params"]:
                    parameter.requires_grad_(False)
        print(
            "Frozen canonical SH radiance; geometry, densification, physics features "
            "and physics network remain trainable.",
            flush=True,
        )
    elif opt.fixed_gaussian_geometry:
        # Freeze topology, opacity, geometry, exposure and Gaussian physics
        # features. SH colours and the separate physics network remain trainable.
        _apply_appearance_only_lrs(gaussians)
        for group in gaussians.optimizer.param_groups:
            if group.get("name") not in {"f_dc", "f_rest"}:
                for parameter in group["params"]:
                    parameter.requires_grad_(False)
        print("Fixed Gaussian geometry/features; SH and physics network remain trainable.", flush=True)

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    weights = _loss_weights(opt)
    camera_stack = []
    consensus_cache = ConsensusCache(opt.consensus_refresh_interval, opt.consensus_label_ema)
    consensus_cache.agreement_scale = opt.consensus_agreement_scale
    consensus_cache.depth_consistency = opt.consensus_depth_consistency
    consensus_cache.min_views = opt.consensus_min_views
    consensus_cache.confidence_threshold = opt.consensus_confidence_threshold
    consensus_neighbors = nearest_camera_map(
        train_cameras, opt.consensus_neighbors, opt.consensus_candidates
    ) if opt.degraded_only else {}
    crossview_neighbors = nearest_camera_map(
        train_cameras, opt.crossview_neighbors, max(opt.crossview_neighbors, opt.consensus_candidates)
    ) if opt.crossview_enabled else {}
    crossview_log = Path(dataset.model_path) / "crossview_diagnostics.jsonl"
    perceptual_model = _lpips_vgg(background.device) if opt.lambda_consensus_perceptual > 0 else None
    degradation_descriptor = build_descriptor(
        opt.descriptor_mode, opt.descriptor_checkpoint, background.device
    )
    onerestore_teacher = build_onerestore_teacher(
        opt.onerestore_teacher_enabled,
        opt.onerestore_embedder_checkpoint,
        opt.onerestore_restorer_checkpoint,
        background.device,
    )
    ema_loss = 0.0
    progress = tqdm(range(first_iteration + 1, opt.iterations + 1), desc="Training")
    save_iterations = set(args.save_iterations + [opt.iterations])
    test_iterations = set(args.test_iterations)
    checkpoint_iterations = set(args.checkpoint_iterations)
    particle_pseudo = ParticlePseudoLabelCache(
        refresh_interval=opt.particle_refresh_interval,
        quantile=opt.particle_residual_quantile,
    )
    checkpoint_particle_pseudo = (
        particle_pseudo if len(physics_models) == 1 else
        ParticlePseudoLabelCache(opt.particle_refresh_interval, opt.particle_residual_quantile)
    )
    gaussian_bank = GaussianObservationBank(
        quantile=opt.gaussian_consensus_quantile,
        min_views=opt.gaussian_consensus_min_views,
        mad_scale=opt.gaussian_consensus_outlier_mad,
        ema_momentum=opt.gaussian_consensus_ema_momentum,
        estimator=opt.gaussian_consensus_estimator,
    )
    if restored_particle_buffers is not None:
        particle_pseudo.load_state_dict(restored_particle_buffers)
    active_stage = None
    best_metric_name = str(opt.best_metric).upper()
    best_metric_value = float("inf") if best_metric_name == "LPIPS" else float("-inf")
    for metric_path in (Path(dataset.model_path) / "metrics").glob("test_iteration_*.json"):
        try:
            metric_iteration = int(metric_path.stem.rsplit("_", 1)[-1])
            if metric_iteration > first_iteration:
                continue
            with metric_path.open("r", encoding="utf-8") as handle:
                value = float(json.load(handle)["average"][best_metric_name])
            if best_metric_name == "LPIPS":
                best_metric_value = min(best_metric_value, value)
            else:
                best_metric_value = max(best_metric_value, value)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    write_run_metadata(
        dataset.model_path,
        {
            "feature_channels": opt.physics_feature_channels,
            "num_views": num_views,
            "clean_eval_dir": opt.clean_dir,
            "final_iteration": opt.iterations,
            "densify_reliability_floor": opt.densify_reliability_floor,
            "densify_weight_min": opt.densify_weight_min,
            "densify_weight_max": opt.densify_weight_max,
            "densification_mode": opt.densification_mode,
            "densify_reliability_threshold": opt.densify_reliability_threshold,
            "densify_reliability_gamma": opt.densify_reliability_gamma,
            "geometry_medium_init_iters": opt.geometry_medium_init_iters,
            "particle_init_iters": opt.particle_init_iters,
            "particle_refresh_interval": opt.particle_refresh_interval,
            "particle_mask_dir": opt.particle_mask_dir,
            "teacher_forcing_end_iter": opt.teacher_forcing_end_iter,
            "exposure_optimization": False,
            "degraded_only": opt.degraded_only,
            "illumination_grid_size": opt.illumination_grid_size,
            "illumination_mode": opt.illumination_mode,
            "physics_v2": opt.physics_v2,
            "haze_decomposition_enabled": opt.haze_decomposition_enabled,
            "particle_decomposition_enabled": opt.particle_decomposition_enabled,
            "haze_depth_normalized": opt.haze_depth_normalized,
            "haze_depth_blur_kernel": opt.haze_depth_blur_kernel,
            "haze_depth_max": opt.haze_depth_max,
            "gamma_range": [opt.gamma_min, opt.gamma_max],
            "consensus_neighbors": opt.consensus_neighbors,
            "consensus_candidates": opt.consensus_candidates,
            "consensus_label_ema": opt.consensus_label_ema,
            "alternating_steps": [opt.alternating_gaussian_steps, opt.alternating_physics_steps],
            "canonical_reliability_enabled": opt.canonical_reliability_enabled,
            "canonical_reliability_start_iter": opt.canonical_reliability_start_iter,
            "canonical_reliability_ramp_iters": opt.canonical_reliability_ramp_iters,
            "canonical_reliability_floor": opt.canonical_reliability_floor,
            "canonical_reliability_gamma": opt.canonical_reliability_gamma,
            "canonical_reliability_max_weight": opt.canonical_reliability_max_weight,
            "gaussian_consensus_enabled": opt.gaussian_consensus_enabled,
            "gaussian_consensus_start": opt.gaussian_consensus_start,
            "gaussian_consensus_end": opt.gaussian_consensus_end,
            "gaussian_consensus_quantile": opt.gaussian_consensus_quantile,
            "lambda_gaussian_canonical": opt.lambda_gaussian_canonical,
            "gaussian_consensus_schedule_switch_iter": opt.gaussian_consensus_schedule_switch_iter,
            "gaussian_consensus_quantile_stage2": opt.gaussian_consensus_quantile_stage2,
            "lambda_gaussian_canonical_stage2": opt.lambda_gaussian_canonical_stage2,
            "physics_freeze_iter": opt.physics_freeze_iter,
            "gaussian_appearance_only_start_iter": opt.gaussian_appearance_only_start_iter,
            "fixed_gaussian_geometry": opt.fixed_gaussian_geometry,
            "gaussian_init_ply": opt.gaussian_init_ply,
            "physics_init_path": opt.physics_init_path,
            "freeze_canonical_gaussians": opt.freeze_canonical_gaussians,
            "freeze_canonical_radiance": opt.freeze_canonical_radiance,
            "allow_physics_densification_when_frozen": opt.allow_physics_densification_when_frozen,
            "dcp_residual_teacher_enabled": opt.dcp_residual_teacher_enabled,
            "dcp_residual_teacher_start_iter": opt.dcp_residual_teacher_start_iter,
            "dcp_residual_max_rgb_delta": opt.dcp_residual_max_rgb_delta,
            "lambda_dcp_residual_anchor": opt.lambda_dcp_residual_anchor,
            "crossview": {key: value for key, value in vars(opt).items()
                          if key.startswith("crossview_") or key.startswith("lambda_crossview_")},
            "descriptor_mode": opt.descriptor_mode,
            "descriptor_checkpoint": opt.descriptor_checkpoint,
            "descriptor_start_iter": opt.descriptor_start_iter,
            "descriptor_ramp_iters": opt.descriptor_ramp_iters,
            "descriptor_interval": opt.descriptor_interval,
            "descriptor_weights": [
                opt.lambda_descriptor_relative,
                opt.lambda_descriptor_redegradation,
                opt.lambda_descriptor_purity,
            ],
            "onerestore_teacher_enabled": opt.onerestore_teacher_enabled,
            "onerestore_embedder_checkpoint": opt.onerestore_embedder_checkpoint,
            "onerestore_restorer_checkpoint": opt.onerestore_restorer_checkpoint,
            "onerestore_teacher_start_iter": opt.onerestore_teacher_start_iter,
            "onerestore_teacher_ramp_iters": opt.onerestore_teacher_ramp_iters,
            "lambda_onerestore_teacher": opt.lambda_onerestore_teacher,
            "onerestore_teacher_confidence_threshold": opt.onerestore_teacher_confidence_threshold,
            "clean_oracle_dir": opt.clean_oracle_dir,
            "lambda_clean_oracle": opt.lambda_clean_oracle,
            "degraded_analytic_prior_dir": opt.degraded_analytic_prior_dir,
            "lambda_degraded_analytic_prior": opt.lambda_degraded_analytic_prior,
            "multi_condition_dirs": [str(path) for path in condition_roots],
        },
    )

    for iteration in progress:
        gaussian_schedule_stage2 = (
            int(opt.gaussian_consensus_schedule_switch_iter) > 0
            and iteration > int(opt.gaussian_consensus_schedule_switch_iter)
        )
        effective_gaussian_quantile = (
            float(opt.gaussian_consensus_quantile_stage2)
            if gaussian_schedule_stage2
            else float(opt.gaussian_consensus_quantile)
        )
        effective_lambda_gaussian_canonical = (
            float(opt.lambda_gaussian_canonical_stage2)
            if gaussian_schedule_stage2
            else float(opt.lambda_gaussian_canonical)
        )
        gaussian_bank.quantile = effective_gaussian_quantile
        gaussian_schedule_just_switched = (
            int(opt.gaussian_consensus_schedule_switch_iter) > 0
            and iteration == int(opt.gaussian_consensus_schedule_switch_iter) + 1
        )
        if gaussian_schedule_just_switched:
            print(
                f"\n[ITER {iteration}] Gaussian teacher schedule switched: "
                f"quantile={effective_gaussian_quantile:.2f}, "
                f"lambda={effective_lambda_gaussian_canonical:.4f}",
                flush=True,
            )
        condition_index = (iteration - 1) % max(1, len(condition_roots))
        physics = physics_models[condition_index]
        active_observed_dir = str(condition_image_dirs[condition_index]) if condition_image_dirs else ""
        gaussians.update_learning_rate(iteration)
        stage = training_stage(
            iteration, opt.geometry_medium_init_iters, opt.particle_init_iters
        )
        _apply_stage_geometry_lrs(gaussians, opt, stage)
        appearance_only = (
            int(opt.gaussian_appearance_only_start_iter) > 0
            and iteration > int(opt.gaussian_appearance_only_start_iter)
        )
        physics_frozen = (
            int(opt.physics_freeze_iter) > 0
            and iteration > int(opt.physics_freeze_iter)
        )
        if opt.freeze_canonical_gaussians:
            _apply_physics_feature_only_lrs(gaussians)
        elif opt.freeze_canonical_radiance:
            _apply_fixed_radiance_lrs(gaussians)
        elif opt.dcp_residual_teacher_enabled:
            _apply_dcp_residual_teacher_lrs(gaussians, opt, iteration, physics_frozen)
        elif appearance_only or opt.fixed_gaussian_geometry:
            _apply_appearance_only_lrs(gaussians)
        if opt.freeze_canonical_gaussians:
            # Step both optimizers: the Gaussian optimizer owns only the two
            # active physics-feature groups in this mode.
            gaussian_phase = physics_phase = True
        elif appearance_only:
            gaussian_phase, physics_phase = True, False
        elif opt.degraded_only:
            cycle = max(1, opt.alternating_gaussian_steps + opt.alternating_physics_steps)
            gaussian_phase = ((iteration - 1) % cycle) < opt.alternating_gaussian_steps
            physics_phase = not gaussian_phase
        else:
            gaussian_phase = physics_phase = True
        if physics_frozen and any(parameter.requires_grad for parameter in physics_models.parameters()):
            for parameter in physics_models.parameters():
                parameter.requires_grad_(False)
            print(
                f"\n[ITER {iteration}] physics frozen; Gaussian appearance-only refinement enabled={appearance_only}",
                flush=True,
            )
        if stage != active_stage:
            active_stage = stage
        sh_interval = 4000 if opt.degraded_only else 1000
        if iteration % sh_interval == 0:
            gaussians.oneupSHdegree()
        if not camera_stack:
            camera_stack = scene.getTrainCameras().copy()
        camera = camera_stack.pop(random.randrange(len(camera_stack)))
        if iteration - 1 == args.debug_from:
            pipe.debug = True

        bg = torch.rand(3, device="cuda") if opt.random_background else background
        package = render(
            camera, gaussians, pipe, bg,
            separate_sh=SPARSE_ADAM_AVAILABLE, render_features=True)
        observed = load_optional_reference(
            active_observed_dir, camera.image_name, camera.original_image.shape[-2:], torch.device("cuda")
        ) if active_observed_dir else camera.original_image.cuda()
        if observed is None:
            raise FileNotFoundError(f"Missing condition image {camera.image_name} in {active_observed_dir}")
        observed = observed.squeeze(0)
        view_index = min(int(camera.uid), num_views - 1)
        eta_l, eta_t, eta_m = trust_weights(
            iteration,
            opt.geometry_medium_init_iters,
            opt.particle_init_iters,
            opt.low_light_trust_start_iter,
            opt.transmission_trust_ramp_iters,
            opt.low_light_trust_ramp_iters,
        )
        blend = particle_strength(
            iteration,
            opt.geometry_medium_init_iters,
            opt.particle_init_iters,
            stage,
        )
        particle_mask_gt = load_optional_reference(
            "" if opt.degraded_only else opt.particle_mask_dir, camera.image_name, observed.shape[-2:], observed.device,
            grayscale=True,
        )
        particle_buffer = None
        particle_mask_est = None
        particle_key = f"c{condition_index}:{camera.image_name}"
        particle_branch_active = (
            iteration > int(opt.nimbus_plm_start_iter)
            and (int(opt.nimbus_plm_end_iter) <= 0 or iteration <= int(opt.nimbus_plm_end_iter))
        ) if opt.nimbus_plm_enabled else (stage != "geometry_medium_init" or opt.degraded_only)
        particle_branch_active = bool(opt.particle_decomposition_enabled) and particle_branch_active
        if particle_branch_active:
            cached = particle_pseudo.cached(particle_key, observed.device, observed.dtype)
            if cached is not None and not particle_pseudo.needs_refresh(particle_key, iteration):
                if opt.degraded_only and not opt.nimbus_plm_enabled:
                    particle_buffer = torch.zeros_like(cached.residual)
                    particle_mask_est = cached.mask
                else:
                    particle_buffer = cached.residual

        output, raster = run_physics(
            physics, package, observed, view_index, opt.physics_feature_channels,
            particle_enabled=particle_branch_active,
            particle_strength=blend,
            particle_residual=particle_buffer,
            particle_mask=(particle_mask_gt if particle_mask_gt is not None else particle_mask_est),
            low_light_trust_weight=eta_l,
            transmission_trust_weight=eta_t,
            particle_trust_weight=eta_m,
            camera=camera,
        )
        if particle_branch_active and particle_buffer is None:
            pseudo = particle_pseudo.get(
                particle_key, iteration, observed, output.continuous_degraded
            )
            output, raster = run_physics(
                physics, package, observed, view_index, opt.physics_feature_channels,
                particle_enabled=True,
                particle_strength=blend,
                particle_residual=(
                    pseudo.residual if opt.nimbus_plm_enabled
                    else (torch.zeros_like(pseudo.residual) if opt.degraded_only else pseudo.residual)
                ),
                particle_mask=(particle_mask_gt if particle_mask_gt is not None else pseudo.mask),
                low_light_trust_weight=eta_l,
                transmission_trust_weight=eta_t,
                particle_trust_weight=eta_m,
                camera=camera,
            )
        canonical_reliability_mean = None
        isolate_dcp_canonical = (
            bool(opt.dcp_residual_teacher_enabled)
            and output.clean_rgb.requires_grad
        )
        canonical_reliability_active = (
            bool(opt.canonical_reliability_enabled)
            and iteration >= int(opt.canonical_reliability_start_iter)
            and output.clean_rgb.requires_grad
            and not isolate_dcp_canonical
        )
        if isolate_dcp_canonical:
            # The degraded reconstruction trains the physics modules and the
            # rasterized physics embeddings, but must not pull the imported
            # DCP canonical back toward haze/rain/snow.  The SH-DC parameter is
            # updated only by the direct Gaussian-teacher and anchor losses.
            output.clean_rgb.register_hook(lambda grad: torch.zeros_like(grad))
        elif canonical_reliability_active:
            # This hook changes only the gradient flowing back into the
            # canonical Gaussian rendering. Gradients accumulated by the
            # downstream illumination/haze/particle modules remain unchanged.
            reliability = output.pixel_reliability.detach().clamp(0.0, 1.0)
            reliability = reliability.clamp_min(float(opt.canonical_reliability_floor))
            reliability = reliability.pow(float(opt.canonical_reliability_gamma))
            reliability = reliability / reliability.mean().clamp_min(1e-6)
            reliability = reliability.clamp_max(float(opt.canonical_reliability_max_weight))
            ramp = min(
                1.0,
                max(
                    0.0,
                    (iteration - int(opt.canonical_reliability_start_iter) + 1)
                    / max(1, int(opt.canonical_reliability_ramp_iters)),
                ),
            )
            canonical_weight = (1.0 + ramp * (reliability - 1.0)).detach()
            canonical_reliability_mean = canonical_weight.mean().detach()
            output.clean_rgb.register_hook(lambda grad: grad * canonical_weight)
        gq_refresh_interval = (
            int(opt.gaussian_consensus_refresh_interval_pre_densify)
            if iteration < int(opt.gaussian_consensus_ema_start)
            else int(opt.gaussian_consensus_refresh_interval)
        )
        refresh_gq = (
            opt.gaussian_consensus_enabled
            and iteration >= int(opt.gaussian_consensus_start)
            and (
                int(opt.gaussian_consensus_end) <= 0
                or iteration <= int(opt.gaussian_consensus_end)
            )
            and (
                gaussian_schedule_just_switched
                or
                gaussian_bank.targets is None
                or gaussian_bank.targets.rgb.shape[0] != gaussians.get_xyz.shape[0]
                or iteration - gaussian_bank.targets.iteration >= gq_refresh_interval
            )
        )
        if refresh_gq:
            _refresh_gaussian_observation_bank(
                gaussian_bank, train_cameras, gaussians, physics, pipe, background,
                opt.physics_feature_channels, opt, iteration,
            )
        consensus_label = (
            _get_consensus_label(
                consensus_cache, consensus_neighbors, camera, raster, iteration,
                opt.consensus_start_iter, gaussians, physics, pipe, background,
                opt.physics_feature_channels, active_observed_dir,
            ) if opt.degraded_only and not opt.crossview_enabled else None
        )
        persistent_edge_map = None
        transient_vote = None
        if consensus_label is not None and (opt.persistent_edge_enabled or opt.transient_voting_enabled):
            persistent_edge_map, transient_vote = _persistent_and_transient_maps(
                consensus_label, output, observed
            )
            if opt.transient_voting_enabled:
                output.particle_mask = torch.maximum(output.particle_mask, transient_vote)
                output.pixel_reliability = output.pixel_reliability * (1.0 - transient_vote)
        effective_attenuation_gt = load_optional_reference(
            "" if opt.degraded_only else opt.effective_attenuation_dir,
            camera.image_name,
            output.final.shape[-2:],
            output.final.device,
            grayscale=True,
        )
        transmission_gt = load_optional_reference(
            "" if opt.degraded_only else opt.transmission_dir, camera.image_name, output.final.shape[-2:], output.final.device,
            grayscale=True,
        )
        tau_gt = load_optional_reference(
            "" if opt.degraded_only else opt.tau_dir, camera.image_name, output.final.shape[-2:], output.final.device,
            grayscale=True,
        )
        path_radiance_gt = load_optional_reference(
            "" if opt.degraded_only else opt.path_radiance_dir, camera.image_name, output.final.shape[-2:], output.final.device
        )
        beta_gt = (
            torch.tensor(opt.beta_gt, device=output.final.device)
            if (not opt.degraded_only and opt.beta_gt >= 0.0) else None
        )
        airlight_values = (opt.airlight_gt_r, opt.airlight_gt_g, opt.airlight_gt_b)
        airlight_gt = (
            torch.tensor(airlight_values, device=output.final.device)
            if (not opt.degraded_only and all(value >= 0.0 for value in airlight_values)) else None
        )
        batch = make_training_batch(
            observed,
            raster.depth,
            effective_attenuation_gt=effective_attenuation_gt,
            transmission_gt=transmission_gt,
            tau_gt=tau_gt,
            path_radiance_gt=path_radiance_gt,
            clean_pseudo_gt=(consensus_label.clean if consensus_label is not None else None),
            clean_pseudo_confidence=(consensus_label.confidence if consensus_label is not None else None),
            particle_mask_gt=particle_mask_gt,
            beta_gt=beta_gt,
            airlight_gt=airlight_gt,
        )
        loss, terms = compute_physics_losses(
            output, batch, weights,
            prediction_weight=physics_prediction_weight(iteration, opt.teacher_forcing_end_iter),
            particle_pixel_weight=opt.particle_pixel_weight,
            pseudo_clean_min_attenuation=opt.pseudo_clean_min_attenuation,
            illumination_anchor=opt.illumination_anchor,
            perceptual_model=perceptual_model,
            clean_pseudo_scale=(
                min(1.0, max(0.0, (iteration - (
                    int(opt.consensus_supervision_start_iter)
                    if int(opt.consensus_supervision_start_iter) > 0 else int(opt.consensus_start_iter)
                )) / max(1, opt.consensus_ramp_iters)))
                if opt.degraded_only else 1.0
            ),
        )
        supervision_start = (
            int(opt.consensus_supervision_start_iter)
            if int(opt.consensus_supervision_start_iter) > 0 else int(opt.consensus_start_iter)
        )
        if (
            consensus_label is not None
            and float(opt.lambda_frequency_consensus) > 0.0
            and iteration >= supervision_start
        ):
            frequency_loss = _frequency_consensus_loss(
                output.clean_rgb, consensus_label.clean, consensus_label.confidence
            )
            scale = min(1.0, max(0.0, (iteration - supervision_start + 1) / max(1, opt.consensus_ramp_iters)))
            loss = loss + float(opt.lambda_frequency_consensus) * scale * frequency_loss
            terms["frequency_consensus"] = frequency_loss.detach()
        if opt.csm_enabled and float(opt.csm_lambda_tv) > 0.0:
            csm_tv = physics.csm_fog.voxel_grid.compute_tv_loss()
            loss = loss + float(opt.csm_lambda_tv) * csm_tv
            terms["csm_tv"] = csm_tv.detach()
        if int(opt.dcp_init_end_iter) > 0 and iteration <= int(opt.dcp_init_end_iter):
            dcp = dark_channel_prior(output.clean_rgb.clamp(0.0, 1.0), use_soft_min=True).abs().mean()
            loss = loss + float(opt.lambda_dcp_init) * dcp
            terms["dcp_init"] = dcp.detach()
        if canonical_reliability_mean is not None:
            terms["canonical_reliability"] = canonical_reliability_mean
        teacher_active = (
            onerestore_teacher is not None
            and iteration >= int(opt.onerestore_teacher_start_iter)
        )
        if teacher_active:
            # Deliberately run the frozen teacher online. The pseudo-clean
            # target exists only for this iteration and is never written out.
            # Clone outside inference_mode so autograd may safely save this
            # frozen target while differentiating the canonical prediction.
            teacher_target = onerestore_teacher(observed).detach().clone()
            confidence = output.pixel_reliability.detach().clamp(0.0, 1.0)
            threshold = float(opt.onerestore_teacher_confidence_threshold)
            confidence = ((confidence - threshold) / max(1e-6, 1.0 - threshold)).clamp(0.0, 1.0)
            pixel_loss = F.smooth_l1_loss(
                output.clean_rgb, teacher_target, reduction="none", beta=0.05
            ).mean(dim=1, keepdim=True)
            teacher_loss = (pixel_loss * confidence).sum() / confidence.sum().clamp_min(1.0)
            teacher_scale = min(
                1.0,
                max(
                    0.0,
                    (iteration - int(opt.onerestore_teacher_start_iter) + 1)
                    / max(1, int(opt.onerestore_teacher_ramp_iters)),
                ),
            )
            loss = loss + float(opt.lambda_onerestore_teacher) * teacher_scale * teacher_loss
            terms["onerestore_teacher"] = teacher_loss.detach()
            terms["onerestore_teacher_confidence"] = confidence.mean().detach()
        if float(opt.lambda_clean_oracle) > 0.0:
            clean_target = load_optional_reference(
                opt.clean_oracle_dir, camera.image_name, output.clean_rgb.shape[-2:], output.clean_rgb.device
            )
            if clean_target is None:
                raise FileNotFoundError(f"Missing clean oracle target for {camera.image_name}")
            clean_l1 = F.l1_loss(output.clean_rgb, clean_target)
            clean_dssim = 1.0 - ssim(output.clean_rgb.squeeze(0), clean_target.squeeze(0))
            clean_oracle_loss = (
                (1.0 - float(opt.lambda_clean_oracle_dssim)) * clean_l1
                + float(opt.lambda_clean_oracle_dssim) * clean_dssim
            )
            loss = loss + float(opt.lambda_clean_oracle) * clean_oracle_loss
            terms["clean_oracle"] = clean_oracle_loss.detach()
        if float(opt.lambda_degraded_analytic_prior) > 0.0:
            analytic_target = load_optional_reference(
                opt.degraded_analytic_prior_dir,
                camera.image_name,
                output.clean_rgb.shape[-2:],
                output.clean_rgb.device,
            )
            if analytic_target is None:
                raise FileNotFoundError(
                    f"Missing degraded analytic prior target for {camera.image_name}"
                )
            analytic_l1 = F.l1_loss(output.clean_rgb, analytic_target)
            analytic_dssim = 1.0 - ssim(
                output.clean_rgb.squeeze(0), analytic_target.squeeze(0)
            )
            analytic_loss = (
                (1.0 - float(opt.lambda_degraded_analytic_prior_dssim)) * analytic_l1
                + float(opt.lambda_degraded_analytic_prior_dssim) * analytic_dssim
            )
            loss = loss + float(opt.lambda_degraded_analytic_prior) * analytic_loss
            terms["degraded_analytic_prior"] = analytic_loss.detach()
        gaussian_teacher_active = (
            opt.gaussian_consensus_enabled
            and gaussian_bank.targets is not None
            and (
                int(opt.gaussian_consensus_end) <= 0
                or iteration <= int(opt.gaussian_consensus_end)
            )
        )
        if gaussian_teacher_active:
            gq_scale = min(
                1.0,
                max(0.0, (iteration - int(opt.gaussian_consensus_start) + 1) / max(1, opt.gaussian_consensus_ramp_iters)),
            )
            # The current view is explicitly excluded when it participated in
            # the refresh; otherwise it was never in the teacher observations.
            gq_loss = gaussian_bank.loss(gaussians.get_features_dc, int(camera.uid))
            loss = loss + effective_lambda_gaussian_canonical * gq_scale * gq_loss
            terms["gaussian_canonical"] = gq_loss.detach()

        if (
            opt.dcp_residual_teacher_enabled
            and iteration >= int(opt.dcp_residual_teacher_start_iter)
        ):
            # SH2RGB(f_dc) is affine, so anchoring the DC RGB directly gives a
            # clear, scene-independent interpretation of the regularizer.
            sh_c0 = 0.28209479177387814
            current_dc_rgb = gaussians.get_features_dc[:, 0, :] * sh_c0 + 0.5
            anchor_dc_rgb = dcp_anchor_dc[:, 0, :] * sh_c0 + 0.5
            dcp_anchor_loss = F.l1_loss(current_dc_rgb, anchor_dc_rgb)
            loss = loss + float(opt.lambda_dcp_residual_anchor) * dcp_anchor_loss
            terms["dcp_residual_anchor"] = dcp_anchor_loss.detach()

        descriptor_due = (
            degradation_descriptor is not None
            and iteration >= int(opt.descriptor_start_iter)
            and iteration % max(1, int(opt.descriptor_interval)) == 0
        )
        if descriptor_due:
            descriptor_scale = min(
                1.0,
                max(0.0, (iteration - int(opt.descriptor_start_iter) + 1) / max(1, int(opt.descriptor_ramp_iters))),
            )
            dloss = descriptor_losses(
                degradation_descriptor,
                output.clean_rgb,
                output.low_light_clean,
                output.continuous_degraded,
                output.final,
                observed,
                margin=opt.descriptor_margin,
            )
            loss = (
                loss
                + descriptor_scale * float(opt.lambda_descriptor_relative) * dloss.relative
                + descriptor_scale * float(opt.lambda_descriptor_redegradation) * dloss.redegradation
                + descriptor_scale * float(opt.lambda_descriptor_purity) * dloss.purity
            )
            terms["descriptor_relative"] = dloss.relative.detach()
            terms["descriptor_redegradation"] = dloss.redegradation.detach()
            terms["descriptor_purity"] = dloss.purity.detach()

        crossview_stats = None
        if (opt.crossview_enabled and iteration >= opt.crossview_start_iter
                and (iteration - opt.crossview_start_iter) % opt.crossview_interval == 0):
            crossview_loss, crossview_stats = crossview_objective(
                camera, crossview_neighbors.get(camera.image_name, []), output, raster,
                package, observed, gaussians, physics, pipe, bg, opt, physics_phase,
                separate_sh=SPARSE_ADAM_AVAILABLE, observed_dir=active_observed_dir)
            ramp = min(1.0, (iteration - opt.crossview_start_iter + 1) / max(1, opt.crossview_ramp_iters))
            loss = loss + ramp * crossview_loss
            crossview_stats.update(iteration=iteration, camera=camera.image_name,
                                   phase="physics" if physics_phase else "gaussian", ramp=ramp)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite cross-view training loss at {iteration}")

        gaussians.optimizer.zero_grad(set_to_none=True)
        physics_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if crossview_stats is not None:
            if physics_phase:
                gradients = [p.grad for p in physics.parameters() if p.grad is not None]
                if any(not torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError(f"Nonfinite physics gradient at {iteration}")
                crossview_stats["physics_grad_norm"] = float(torch.stack(
                    [g.detach().square().sum() for g in gradients]).sum().sqrt()) if gradients else 0.0
            with crossview_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(crossview_stats) + "\n")
            if iteration == first_iteration + 1 or iteration % 25 == 0:
                print(f"\n[ITER {iteration}] crossview {crossview_stats}", flush=True)

        with torch.no_grad():
            radii = package["radii"]
            visibility = package["visibility_filter"]
            if (gaussian_phase and not opt.fixed_gaussian_geometry
                    and (
                        not opt.freeze_canonical_gaussians
                        or opt.allow_physics_densification_when_frozen
                    )
                    and iteration < opt.densify_until_iter):
                gaussians.max_radii2D[visibility] = torch.maximum(
                    gaussians.max_radii2D[visibility], radii[visibility]
                )
                gaussians.add_densification_stats(package["viewspace_points"], visibility)
                if stage != "geometry_medium_init" and opt.densification_mode == "physics":
                    view_reliability = project_reliability_to_gaussians(
                        camera, gaussians.get_xyz, output.pixel_reliability, radii
                    )
                    gaussians.add_physics_trust_stats(view_reliability, visibility)
                # Offset scheduled maintenance by one iteration.  With the
                # default 20 Gaussian / 5 physics alternating cycle, exact
                # multiples of 100 are physics-only iterations; the old
                # condition therefore prevented densification from ever
                # running.  501, 601, ... are Gaussian-phase iterations.
                densify_due = (
                    iteration > opt.densify_from_iter
                    and (iteration - (opt.densify_from_iter + 1)) % opt.densification_interval == 0
                )
                if densify_due:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    densify_threshold = opt.densify_grad_threshold
                    densify_kwargs = {}
                    if stage == "geometry_medium_init":
                        densify_threshold *= opt.stage1_densify_threshold_multiplier
                    else:
                        error_map = (output.final - observed.unsqueeze(0)).abs().mean(dim=1)
                        edge_active = (
                            opt.persistent_edge_enabled
                            and persistent_edge_map is not None
                            and iteration >= int(opt.persistent_edge_start_iter)
                            and (int(opt.persistent_edge_end_iter) <= 0 or iteration <= int(opt.persistent_edge_end_iter))
                        )
                        if edge_active:
                            error_map = error_map * (
                                1.0 + float(opt.persistent_edge_boost) * persistent_edge_map.squeeze(1)
                            )
                            if transient_vote is not None:
                                error_map = error_map * (1.0 - transient_vote.squeeze(1))
                            densify_threshold *= float(opt.persistent_edge_threshold_multiplier)
                        densify_kwargs = {
                            "viewpoint_cam": camera,
                            "error_map": error_map,
                            "physics_reliability": (
                                gaussians.get_average_physics_trust()
                                if opt.densification_mode == "physics" else None
                            ),
                        }
                    gaussians.densify_and_prune(
                        densify_threshold,
                        float(opt.densify_prune_min_opacity),
                        scene.cameras_extent,
                        size_threshold,
                        radii,
                        **densify_kwargs,
                    )
                if iteration > 1 and (iteration - 1) % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                if gaussian_phase:
                    gaussians.optimizer.step()
                    if opt.dcp_residual_teacher_enabled:
                        sh_c0 = 0.28209479177387814
                        max_sh_delta = float(opt.dcp_residual_max_rgb_delta) / sh_c0
                        lower = dcp_anchor_dc - max_sh_delta
                        upper = dcp_anchor_dc + max_sh_delta
                        gaussians.get_features_dc.copy_(
                            torch.maximum(
                                lower,
                                torch.minimum(upper, gaussians.get_features_dc),
                            )
                        )
                if physics_phase:
                    physics_optimizer.step()

            ema_loss = 0.4 * float(loss) + 0.6 * ema_loss
            if iteration % 10 == 0:
                progress.set_postfix(
                    loss=f"{ema_loss:.6f}",
                    points=gaussians.get_xyz.shape[0],
                    stage=stage,
                    phase=("gaussian" if gaussian_phase else "physics"),
                )

            if iteration in test_iterations:
                metrics = _validate(
                    scene,
                    gaussians,
                    physics,
                    pipe,
                    background,
                    opt.physics_feature_channels,
                    opt.clean_dir,
                    iteration,
                )
                print(f"\n[ITER {iteration}] validation {metrics}", flush=True)
                if best_metric_name in metrics:
                    candidate = metrics[best_metric_name]
                    improved = candidate < best_metric_value if best_metric_name == "LPIPS" else candidate > best_metric_value
                    if improved:
                        best_metric_value = candidate
                        if opt.save_best_checkpoint:
                            torch.save(
                                _checkpoint_payload(iteration, gaussians, (physics_models if len(physics_models) > 1 else physics), physics_optimizer, opt.physics_feature_channels, num_views, stage, checkpoint_particle_pseudo),
                                Path(dataset.model_path) / "best.pth",
                            )
                            print(f"[ITER {iteration}] new best {best_metric_name}={candidate:.6f} -> best.pth", flush=True)

            if iteration in save_iterations:
                scene.save(iteration)
                if len(physics_models) == 1:
                    save_physics_state(
                        Path(dataset.model_path) / "physics" / f"iteration_{iteration}.pt",
                        physics, iteration, opt.physics_feature_channels, physics_optimizer,
                        particle_pseudo.state_dict(),
                    )
                else:
                    physics_dir = Path(dataset.model_path) / "physics"
                    physics_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "iteration": iteration,
                        "physics": physics_models.state_dict(),
                        "optimizer": physics_optimizer.state_dict(),
                        "condition_roots": [str(path) for path in condition_roots],
                        "particle_buffers": {},
                    }, physics_dir / f"iteration_{iteration}.pt")

            if iteration in checkpoint_iterations:
                torch.save(
                    _checkpoint_payload(iteration, gaussians, (physics_models if len(physics_models) > 1 else physics), physics_optimizer, opt.physics_feature_channels, num_views, stage, checkpoint_particle_pseudo),
                    Path(dataset.model_path) / f"chkpnt{iteration}.pth",
                )


def build_parser() -> tuple[ArgumentParser, ModelParams, OptimizationParams, PipelineParams]:
    parser = ArgumentParser(description="End-to-end Physics-aware Modular 3DGS training")
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--test_iterations", nargs="*", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="*", type=int, default=[7000, 30000])
    parser.add_argument("--checkpoint_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default="")
    return parser, model, optimization, pipeline


def main() -> None:
    parser, model, optimization, pipeline = build_parser()
    args = parse_configured_args(parser)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(model.extract(args), optimization.extract(args), pipeline.extract(args), args)
    print("Training complete.")


if __name__ == "__main__":
    main()
