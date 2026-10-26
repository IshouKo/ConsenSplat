from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ParticlePseudoLabel:
    mask: torch.Tensor
    residual: torch.Tensor
    iteration: int


def make_particle_pseudo_label(
    observed: torch.Tensor,
    continuous: torch.Tensor,
    quantile: float = 0.95,
) -> ParticlePseudoLabel:
    """Create the non-learned NimbusGS positive residual and robust strength."""
    observed = observed if observed.dim() == 4 else observed.unsqueeze(0)
    continuous = continuous if continuous.dim() == 4 else continuous.unsqueeze(0)
    residual = (observed.detach() - continuous.detach()).clamp_min(0.0)
    intensity = residual.mean(dim=1, keepdim=True)
    q = min(max(float(quantile), 0.0), 1.0)
    scale = torch.quantile(intensity.flatten(1), q, dim=1).view(-1, 1, 1, 1)
    strength = (intensity >= scale.clamp_min(1e-6)).to(intensity.dtype)
    return ParticlePseudoLabel(mask=strength.detach(), residual=residual.detach(), iteration=0)


class ParticlePseudoLabelCache:
    """Keep per-view pseudo-labels fixed between scheduled refreshes."""

    def __init__(
        self,
        refresh_interval: int = 100,
        quantile: float = 0.95,
    ) -> None:
        self.refresh_interval = max(1, int(refresh_interval))
        self.quantile = float(quantile)
        self._labels: dict[str | int, ParticlePseudoLabel] = {}

    def needs_refresh(self, view_id: str | int, iteration: int) -> bool:
        cached = self._labels.get(view_id)
        return cached is None or int(iteration) - cached.iteration >= self.refresh_interval

    def cached(
        self, view_id: str | int, device: torch.device, dtype: torch.dtype
    ) -> ParticlePseudoLabel | None:
        value = self._labels.get(view_id)
        if value is None:
            return None
        return ParticlePseudoLabel(
            mask=value.mask.to(device=device, dtype=dtype),
            residual=value.residual.to(device=device, dtype=dtype),
            iteration=value.iteration,
        )

    def get(
        self,
        view_id: str | int,
        iteration: int,
        observed: torch.Tensor,
        continuous: torch.Tensor,
    ) -> ParticlePseudoLabel:
        cached = self._labels.get(view_id)
        should_refresh = self.needs_refresh(view_id, iteration)
        if should_refresh:
            label = make_particle_pseudo_label(
                observed,
                continuous,
                quantile=self.quantile,
            )
            cached = ParticlePseudoLabel(
                mask=label.mask.cpu(),
                residual=label.residual.cpu(),
                iteration=int(iteration),
            )
            self._labels[view_id] = cached
        device = continuous.device
        dtype = continuous.dtype
        return ParticlePseudoLabel(
            mask=cached.mask.to(device=device, dtype=dtype),
            residual=cached.residual.to(device=device, dtype=dtype),
            iteration=cached.iteration,
        )

    def __len__(self) -> int:
        return len(self._labels)

    def state_dict(self) -> dict[str | int, dict[str, torch.Tensor | int]]:
        return {
            view_id: {
                "mask": label.mask.cpu(),
                "residual": label.residual.cpu(),
                "iteration": int(label.iteration),
            }
            for view_id, label in self._labels.items()
        }

    def load_state_dict(self, state: dict) -> None:
        self._labels = {
            view_id: ParticlePseudoLabel(
                mask=value["mask"].cpu(),
                residual=value["residual"].cpu(),
                iteration=int(value["iteration"]),
            )
            for view_id, value in state.items()
        }
