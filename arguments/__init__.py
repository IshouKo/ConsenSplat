#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import os
import sys
import yaml

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            arg_name = key
            if arg_name.startswith("_"):
                shorthand = True
                arg_name = arg_name[1:]
            value_type = type(value)
            default_value = value if not fill_none else None
            if shorthand:
                if value_type == bool:
                    group.add_argument("--" + arg_name, "-" + arg_name[0:1], default=default_value, action="store_true")
                else:
                    group.add_argument("--" + arg_name, "-" + arg_name[0:1], default=default_value, type=value_type)
            else:
                if value_type == bool:
                    group.add_argument("--" + arg_name, default=default_value, action="store_true")
                else:
                    group.add_argument("--" + arg_name, default=default_value, type=value_type)

    def extract(self, args):
        group = GroupParams()
        for key, value in vars(args).items():
            if key in vars(self) or "_" + key in vars(self):
                setattr(group, key, value)
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Dataset Parameters", sentinel)

    def extract(self, args):
        group = super().extract(args)
        group.source_path = os.path.abspath(group.source_path) if group.source_path else ""
        return group

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.4
        self.densification_interval = 100
        self.opacity_reset_interval = 30000
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        self.densify_max_clone_points = 60000
        self.densify_max_split_points = 60000
        self.densify_max_total_points = 1500000
        self.densify_split_factor = 2
        self.densify_prune_min_opacity = 0.005
        self.densify_reliability_floor = 0.3
        self.densify_reliability_threshold = 0.0
        self.densify_reliability_gamma = 1.0
        self.densify_weight_min = 0.5
        self.densify_weight_max = 3.0
        # "physics" = NimbusGS geometry need * Physics Trust; "geometry" =
        # identical geometry need without the Physics Trust reweighting.
        self.densification_mode = "physics"
        self.geometry_medium_init_iters = 4000
        self.particle_init_iters = 2000
        self.particle_refresh_interval = 100
        self.particle_residual_quantile = 0.95
        self.low_light_trust_start_iter = 10000
        self.transmission_trust_ramp_iters = 1000
        self.low_light_trust_ramp_iters = 1000
        self.stage1_scale_lr_multiplier = 0.1
        self.stage1_feature_lr_multiplier = 0.2
        self.stage2_geometry_lr_multiplier = 0.5
        self.stage1_densify_threshold_multiplier = 2.0
        self.teacher_forcing_end_iter = 10000
        self.particle_pixel_weight = 0.05
        self.pseudo_clean_min_attenuation = 0.05
        # SC-RC v1: gate only the rasterized canonical RGB gradient with the
        # detached physics reliability map. Physics branches still receive the
        # unweighted reconstruction gradient.
        self.canonical_reliability_enabled = False
        self.canonical_reliability_start_iter = 4000
        self.canonical_reliability_ramp_iters = 2000
        self.canonical_reliability_floor = 0.10
        self.canonical_reliability_gamma = 2.0
        self.canonical_reliability_max_weight = 3.0

        self.enable_fog = True
        self.fog_resolution = 128
        self.fog_lr = 0.005
        self.airlight_lr = 0.005
        self.lambda_tv = 1.0
        self.lambda_dcp = 1.0
        self.lambda_tau = 0.0
        self.density_scale_init = 4.0
        self.fog_auto_normalize = True
        self.use_suggested_density = True

        self.use_airlight_cnn = False
        self.use_airlight_cnn2 = False
        self.airlight_feature_dim = 64
        self.airlight_cnn_lr = 0.0001
        self.airlight_sh_degree = 0

        self.enable_rain = False
        self.derain_warmup_iters = 2000
        self.rain_diff_thr = 0.05
        self.rain_update_after_warmup = False
        self.rain_update_interval = 200
        self.densify_rain_mask_fill_ratio = 1.0
        self.densify_rain_mask_threshold = 0.5

        self.gt_path = ""
        self.best_metric = "PSNR"
        self.save_best_checkpoint = True

        # Physics-aware restoration branch.
        self.physics_feature_channels = 8
        self.illumination_grid_size = 16
        self.illumination_mode = "decoder"
        self.physics_v2 = False
        # Explicit modular-physics switches.  These are wired into the current
        # PhysicsAware3DGS model; the older enable_fog/enable_rain options above
        # belong to legacy branches and do not control this decomposition.
        self.low_light_decomposition_enabled = True
        self.haze_decomposition_enabled = True
        self.particle_decomposition_enabled = True
        self.csm_enabled = False
        self.csm_resolution = 128
        self.csm_num_samples = 64
        self.csm_density_scale_init = 0.3
        self.csm_airlight_mode = "cnn"
        self.csm_lambda_tv = 0.0
        self.dcp_init_end_iter = 0
        self.lambda_dcp_init = 0.0
        self.nimbus_plm_enabled = False
        self.nimbus_plm_start_iter = 4000
        self.nimbus_plm_end_iter = 0
        self.haze_depth_normalized = False
        self.haze_depth_blur_kernel = 21
        self.haze_depth_max = 0.7
        self.gamma_min = 1.4
        self.gamma_max = 1.9
        self.degraded_only = False
        self.consensus_neighbors = 3
        self.consensus_candidates = 8
        self.consensus_label_ema = 0.0
        self.consensus_refresh_interval = 200
        self.consensus_start_iter = 4000
        self.consensus_agreement_scale = 0.10
        self.consensus_depth_consistency = 0.10
        self.consensus_min_views = 2
        self.consensus_confidence_threshold = 0.50
        self.consensus_ramp_iters = 4000
        self.consensus_supervision_start_iter = 0
        self.persistent_edge_enabled = False
        self.persistent_edge_start_iter = 4000
        self.persistent_edge_end_iter = 8000
        self.persistent_edge_boost = 2.0
        self.persistent_edge_threshold_multiplier = 0.6
        self.transient_voting_enabled = False
        self.lambda_frequency_consensus = 0.0
        self.lambda_consensus_clean = 0.10
        self.lambda_consensus_ssim = 0.0
        self.lambda_consensus_perceptual = 0.0
        self.lambda_consensus_gradient = 0.0
        # V1-GQ: Gaussian-level robust canonical-radiance consensus.
        self.gaussian_consensus_enabled = False
        self.gaussian_consensus_start = 15000
        self.gaussian_consensus_refresh_interval = 5000
        self.gaussian_consensus_refresh_interval_pre_densify = 1000
        self.gaussian_consensus_max_views = 24
        self.gaussian_consensus_max_views_pre_densify = 8
        self.gaussian_consensus_quantile = 0.80
        self.gaussian_consensus_estimator = "quantile"
        self.gaussian_consensus_min_views = 5
        self.gaussian_consensus_outlier_mad = 2.5
        self.gaussian_consensus_min_radius = 1.0
        self.gaussian_consensus_min_transmission = 0.10
        self.gaussian_consensus_min_reliability = 0.35
        self.gaussian_consensus_depth_consistency = 0.15
        self.lambda_gaussian_canonical = 0.05
        self.gaussian_consensus_ramp_iters = 3000
        self.gaussian_consensus_ema_momentum = 0.90
        self.gaussian_consensus_ema_start = 15000
        # In fused training, apply the MV3D canonical target directly to the
        # base Gaussian DC colour instead of learning a detached residual.
        self.mv3d_teacher_updates_base = False
        self.gaussian_consensus_brightness_calibration = False
        self.gaussian_consensus_brightness_gain_limit = 0.15
        self.gaussian_consensus_brightness_min_confidence = 0.50
        # Remove view-dependent exposure from inverse-physics observations
        # before multi-view aggregation.  This is scalar luminance alignment,
        # so the Teacher's chromatic and local corrections are preserved.
        self.gaussian_consensus_per_view_brightness_calibration = False
        self.gaussian_consensus_per_view_gain_limit = 0.60
        self.gaussian_consensus_loss_beta = 1.0
        # ``absolute`` reproduces the original bank. ``residual`` transfers a
        # centre-sampled inverse-image residual. ``alpha_residual`` uses the
        # renderer adjoint to attribute pixel residuals according to each
        # Gaussian's actual transmittance-times-alpha contribution.
        self.gaussian_consensus_target_mode = "absolute"
        self.gaussian_consensus_max_rgb_delta = 0.05
        self.gaussian_consensus_alpha_pixel_clip = 0.15
        self.gaussian_consensus_min_alpha_contribution = 1e-5
        # A4: gauge-fixed, confidence-stratified, cross-view accepted Teacher.
        self.gaussian_consensus_zero_mean_rgb_update = False
        self.gaussian_consensus_high_confidence = 0.70
        self.gaussian_consensus_structure_confidence = 0.35
        self.lambda_gaussian_structure = 0.0
        self.gaussian_consensus_structure_interval = 5
        # A5 optionally removes local exposure/contrast before comparing
        # Teacher and Base edges.  ``gradient`` preserves the A4 behaviour.
        self.gaussian_consensus_structure_mode = "gradient"
        self.gaussian_consensus_structure_local_window = 9
        self.gaussian_consensus_crossview_acceptance = False
        self.gaussian_consensus_acceptance_mode = "redegradation"
        self.gaussian_consensus_acceptance_views = 1
        self.gaussian_consensus_acceptance_min_improvement = 0.0
        self.gaussian_consensus_acceptance_scales = "1.0,0.5,0.25"
        # Optional second-stage teacher schedule. A value <= 0 disables it.
        self.gaussian_consensus_schedule_switch_iter = 0
        self.gaussian_consensus_quantile_stage2 = 0.80
        self.lambda_gaussian_canonical_stage2 = 0.03
        # Optional finite teacher window.  A value <= 0 preserves the legacy
        # behaviour (teacher remains active until the final iteration).
        self.gaussian_consensus_end = 0
        self.lambda_illumination_anchor = 0.02
        self.illumination_anchor = 0.8
        self.alternating_gaussian_steps = 20
        self.alternating_physics_steps = 5
        # Optional late appearance-refinement stage.  Both switches are off by
        # default so existing configurations and checkpoints are unchanged.
        self.physics_freeze_iter = 0
        self.gaussian_appearance_only_start_iter = 0
        # Fixed-geometry pilot retains alternating physics updates. Unlike the
        # legacy appearance-only stage it does NOT turn off the physics phase.
        self.fixed_gaussian_geometry = False
        # Initialize the canonical scene from an existing 3DGS PLY.  The
        # optional freeze mode keeps geometry, opacity, and SH radiance fixed
        # while learning only per-Gaussian physics features and physics heads.
        self.gaussian_init_ply = ""
        self.physics_init_path = ""
        self.freeze_canonical_gaussians = False
        self.freeze_canonical_radiance = False
        self.allow_physics_densification_when_frozen = False
        # DCP-anchored residual Gaussian teacher.  Geometry, opacity and
        # higher-order SH stay fixed; the Gaussian teacher may update only a
        # bounded DC-colour residual around the imported DCP canonical.  The
        # per-Gaussian physics features remain trainable until physics freeze.
        self.dcp_residual_teacher_enabled = False
        self.dcp_residual_teacher_start_iter = 4000
        self.dcp_residual_max_rgb_delta = 0.06
        self.lambda_dcp_residual_anchor = 0.01
        # Single-loop DCP+3DGS / physics / bounded Gaussian-teacher fusion.
        self.fused_training = False
        self.fused_physics_start_iter = 8000
        self.fused_teacher_start_iter = 18000
        self.fused_teacher_ramp_iters = 2000
        self.fused_teacher_refresh_interval = 1000
        # In direct-base mode DCP remains the geometry/high-order appearance
        # anchor, while its SH-DC gradient may fade so MV3D can become the
        # canonical-colour authority.  A small reliable re-degradation cycle
        # gradient can additionally constrain SH-DC through the physics path.
        self.mv3d_dcp_color_anchor_final_weight = 1.0
        self.mv3d_dcp_color_anchor_fade_iters = 0
        self.mv3d_redegradation_cycle_base_weight = 0.0
        # Search pilots can keep metrics/checkpoints without writing a large
        # final PLY and multi-GB physics artifact for every rejected branch.
        self.pilot_skip_final_artifacts = False
        self.crossview_enabled = False
        self.crossview_start_iter = 15000
        self.crossview_interval = 1
        self.crossview_ramp_iters = 100
        self.crossview_neighbors = 3
        self.crossview_samples = 4096
        self.crossview_min_views = 2
        self.crossview_depth_tolerance = 0.06
        self.crossview_min_information = 0.01
        self.crossview_huber_delta = 0.03
        self.crossview_irls_steps = 3
        self.lambda_crossview_prediction = 0.1
        self.lambda_crossview_appearance = 0.02
        self.physics_lr = 0.0002
        self.physics_weight_decay = 0.0001
        self.beta_init = 0.08
        self.beta_gt = -1.0
        self.airlight_gt_r = -1.0
        self.airlight_gt_g = -1.0
        self.airlight_gt_b = -1.0
        # Evaluation-only clean reference directory; never used by training losses.
        self.clean_dir = ""
        self.effective_attenuation_dir = ""
        self.transmission_dir = ""
        self.tau_dir = ""
        self.path_radiance_dir = ""
        self.particle_mask_dir = ""
        self.lambda_image = 1.0
        self.lambda_low_light_attenuation = 0.0
        self.lambda_transmission = 0.0
        self.lambda_tau_physics = 0.0
        self.lambda_path_radiance = 0.0
        self.lambda_clean_pseudo = 0.0
        self.lambda_beta = 0.0
        self.lambda_airlight = 0.0
        self.lambda_depth_tv = 0.001
        self.lambda_low_light_smooth = 0.02
        # Optional degradation-coordinate prior. "onerestore" uses the frozen
        # official CDD-11 embedder; "selfsup" uses intervention-trained L/H/P coordinates.
        self.descriptor_mode = "none"
        self.descriptor_checkpoint = ""
        self.descriptor_start_iter = 15000
        self.descriptor_ramp_iters = 2000
        self.descriptor_interval = 20
        self.descriptor_margin = 0.03
        self.lambda_descriptor_relative = 0.003
        self.lambda_descriptor_redegradation = 0.002
        self.lambda_descriptor_purity = 0.001
        # Diagnostic oracle: explicit scene-specific clean supervision.
        self.clean_oracle_dir = ""
        self.lambda_clean_oracle = 0.0
        self.lambda_clean_oracle_dssim = 0.2
        # Deterministic pseudo-clean supervision derived only from the active
        # degraded images. Its provenance file is checked before training.
        self.degraded_analytic_prior_dir = ""
        self.lambda_degraded_analytic_prior = 0.0
        self.lambda_degraded_analytic_prior_dssim = 0.2
        # Frozen OneRestore online teacher. The restorer is executed in-memory
        # for every active training iteration; no pseudo-clean images are cached.
        self.onerestore_teacher_enabled = False
        self.onerestore_embedder_checkpoint = ""
        self.onerestore_restorer_checkpoint = ""
        self.onerestore_teacher_start_iter = 4000
        self.onerestore_teacher_ramp_iters = 4000
        self.lambda_onerestore_teacher = 0.10
        self.onerestore_teacher_confidence_threshold = 0.30
        # Semicolon-separated scene roots. They share one Gaussian model and
        # use condition-specific physics heads; clean/ and physics_gt/ are never read.
        self.multi_condition_dirs = ""

        super().__init__(parser, "Optimization Parameters")

def _flatten_config(cfg):
    # Full-pipeline YAML files nest train.py parameters under stage2.
    cfg = cfg.get("stage2", cfg)
    flat = {}
    for section in ("dataset", "pipeline", "optimization"):
        values = cfg.get(section, {})
        if values is not None:
            flat.update(values)
    return flat

def _load_yaml_config(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}

def _build_parser():
    parser = ArgumentParser(description="NimbusGS training")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--test_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default="")
    return parser

def get_combined_args(parser):
    cmd_args = parser.parse_args(sys.argv[1:])
    cfg = _flatten_config(_load_yaml_config(getattr(cmd_args, "config", "")))

    for key, value in cfg.items():
        if hasattr(cmd_args, key):
            current = getattr(cmd_args, key)
            if current == parser.get_default(key):
                setattr(cmd_args, key, value)
        else:
            setattr(cmd_args, key, value)

    return cmd_args

def _ensure_config_argument(parser):
    for action in parser._actions:
        if "--config" in action.option_strings:
            return
    parser.add_argument("--config", type=str, default="")

def parse_configured_args(parser):
    _ensure_config_argument(parser)
    args = parser.parse_args(sys.argv[1:])
    cfg = _flatten_config(_load_yaml_config(getattr(args, "config", "")))
    for key, value in cfg.items():
        if hasattr(args, key):
            default_value = parser.get_default(key)
            if getattr(args, key) == default_value:
                setattr(args, key, value)
        else:
            setattr(args, key, value)
    return args

def create_parser():
    parser = _build_parser()
    ModelParams(parser)
    PipelineParams(parser)
    OptimizationParams(parser)
    return parser
