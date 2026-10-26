from __future__ import annotations

import torch


def pixel_reliability(
    effective_attenuation: torch.Tensor,
    transmission: torch.Tensor,
    particle_strength: torch.Tensor,
    attenuation_reference: float = 1.0,
    light_trust_floor: float = 0.3,
    transmission_trust_floor: float = 0.3,
    low_light_weight: float = 1.0,
    transmission_weight: float = 1.0,
    particle_weight: float = 1.0,
) -> torch.Tensor:
    """Detached, stage-aware pixel trust from the three physical factors."""
    light = effective_attenuation.detach()
    light_rel = (light / max(float(attenuation_reference), 1e-6)).clamp(0.0, 1.0)
    light_trust_floor = min(max(float(light_trust_floor), 0.0), 1.0)
    transmission_trust_floor = min(max(float(transmission_trust_floor), 0.0), 1.0)
    light_trust = light_trust_floor + (1.0 - light_trust_floor) * light_rel
    transmission_trust = transmission_trust_floor + (
        1.0 - transmission_trust_floor
    ) * transmission.clamp(0.0, 1.0)
    particle_trust = 1.0 - particle_strength.detach().clamp(0.0, 1.0)
    eta_l = min(max(float(low_light_weight), 0.0), 1.0)
    eta_t = min(max(float(transmission_weight), 0.0), 1.0)
    eta_m = min(max(float(particle_weight), 0.0), 1.0)
    light_on = (1.0 - eta_l) + eta_l * light_trust
    transmission_on = (1.0 - eta_t) + eta_t * transmission_trust.detach()
    particle_on = (1.0 - eta_m) + eta_m * particle_trust
    return (light_on * transmission_on * particle_on).clamp(0.0, 1.0).detach()


def physics_aware_densification_weights(
    geometry_need: torch.Tensor,
    gaussian_reliability: torch.Tensor | None,
    reliability_floor: float = 0.3,
    min_weight: float = 0.5,
    max_weight: float = 3.0,
    reliability_threshold: float = 0.0,
    reliability_gamma: float = 1.0,
) -> torch.Tensor:
    """Combine NimbusGS-style geometry need with detached physics trust."""
    need = geometry_need.detach().view(-1)
    if gaussian_reliability is None:
        raw = need
        gate = None
    else:
        reliability = gaussian_reliability.detach().view(-1)
        if reliability.numel() != need.numel():
            raise ValueError("Physics trust count must match the number of Gaussian weights.")
        reliability_floor = min(max(float(reliability_floor), 0.0), 1.0)
        trust = reliability_floor + (1.0 - reliability_floor) * reliability.clamp(0.0, 1.0)
        trust = trust.pow(max(float(reliability_gamma), 0.0))
        gate = reliability >= min(max(float(reliability_threshold), 0.0), 1.0)
        raw = need * trust

    raw = torch.nan_to_num(raw, nan=1.0, posinf=max_weight, neginf=0.0)
    normalized = raw / raw.mean().clamp_min(1e-6)
    weights = normalized.clamp(float(min_weight), float(max_weight))
    if gate is not None:
        weights = torch.where(gate, weights, torch.zeros_like(weights))
    return weights.detach()
