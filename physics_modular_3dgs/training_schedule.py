from __future__ import annotations


def training_stage(
    iteration: int,
    geometry_medium_init_iters: int,
    particle_init_iters: int,
) -> str:
    if iteration <= int(geometry_medium_init_iters):
        return "geometry_medium_init"
    if iteration <= int(geometry_medium_init_iters) + int(particle_init_iters):
        return "particle_init"
    return "joint"


def particle_strength(
    iteration: int,
    geometry_medium_init_iters: int,
    particle_init_iters: int,
    stage: str,
) -> float:
    if stage == "geometry_medium_init":
        return 0.0
    if stage == "particle_init":
        start = int(geometry_medium_init_iters) + 1
        return min(1.0, max(0.0, (iteration - start + 1) / max(1, particle_init_iters)))
    return 1.0


def physics_prediction_weight(iteration: int, end_iteration: int) -> float:
    """Ramp from GT-physics teacher forcing to fully predicted physics."""
    return min(1.0, max(0.0, float(iteration) / max(1, int(end_iteration))))


def trust_weights(
    iteration: int,
    geometry_medium_init_iters: int,
    particle_init_iters: int,
    low_light_start_iter: int = 10000,
    transmission_ramp_iters: int = 1000,
    low_light_ramp_iters: int = 1000,
) -> tuple[float, float, float]:
    """Return (low-light attenuation, transmission, particle) trust coefficients."""
    stage = training_stage(iteration, geometry_medium_init_iters, particle_init_iters)
    if stage == "geometry_medium_init":
        return 0.0, 0.0, 0.0
    if stage == "particle_init":
        return 0.0, 0.0, 1.0
    joint_start = int(geometry_medium_init_iters) + int(particle_init_iters) + 1
    eta_t = min(1.0, max(0.0, (iteration - joint_start + 1) / max(1, transmission_ramp_iters)))
    eta_l = min(1.0, max(0.0, (iteration - int(low_light_start_iter) + 1) / max(1, low_light_ramp_iters)))
    return eta_l, eta_t, 1.0
