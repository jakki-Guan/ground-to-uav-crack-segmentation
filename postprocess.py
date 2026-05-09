from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class ConnectedComponentPostprocessConfig:
    min_area: int = 0
    max_fill_ratio: float = 1.0
    min_aspect_ratio: float = 1.0
    max_components: int | None = None

    def __post_init__(self):
        if self.min_area < 0:
            raise ValueError("min_area must be >= 0.")
        if not (0.0 < self.max_fill_ratio <= 1.0):
            raise ValueError("max_fill_ratio must be in the range (0, 1].")
        if self.min_aspect_ratio < 1.0:
            raise ValueError("min_aspect_ratio must be >= 1.0.")
        if self.max_components is not None and self.max_components <= 0:
            raise ValueError("max_components must be positive when provided.")

    @property
    def enabled(self) -> bool:
        return (
            self.min_area > 0
            or self.max_fill_ratio < 1.0
            or self.min_aspect_ratio > 1.0
            or self.max_components is not None
        )


def build_postprocess_config(
    min_area: int = 0,
    max_fill_ratio: float = 1.0,
    min_aspect_ratio: float = 1.0,
    max_components: int = 0,
) -> ConnectedComponentPostprocessConfig | None:
    normalized_max_components = max_components if max_components > 0 else None
    config = ConnectedComponentPostprocessConfig(
        min_area=min_area,
        max_fill_ratio=max_fill_ratio,
        min_aspect_ratio=min_aspect_ratio,
        max_components=normalized_max_components,
    )
    return config if config.enabled else None


def filter_connected_components(
    prob_map: np.ndarray,
    threshold: float = 0.5,
    config: ConnectedComponentPostprocessConfig | None = None,
) -> np.ndarray:
    binary_mask = (prob_map > threshold).astype(np.uint8)
    if config is None or not config.enabled or binary_mask.sum() == 0:
        return binary_mask

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    kept_components: list[tuple[float, int]] = []
    for component_id in range(1, component_count):
        x, y, width, height, area = stats[component_id]
        if area < config.min_area:
            continue

        bbox_area = max(int(width) * int(height), 1)
        fill_ratio = float(area) / float(bbox_area)
        if fill_ratio > config.max_fill_ratio:
            continue

        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
        if aspect_ratio < config.min_aspect_ratio:
            continue

        component_mask = labels == component_id
        mean_probability = float(prob_map[component_mask].mean())
        score = mean_probability * float(area) * aspect_ratio
        kept_components.append((score, component_id))

    if not kept_components:
        return np.zeros_like(binary_mask)

    if config.max_components is not None:
        kept_components.sort(reverse=True)
        kept_ids = {component_id for _, component_id in kept_components[: config.max_components]}
    else:
        kept_ids = {component_id for _, component_id in kept_components}

    return np.isin(labels, list(kept_ids)).astype(np.uint8)


def logits_to_binary_mask(
    logits: torch.Tensor,
    threshold: float = 0.5,
    postprocess_config: ConnectedComponentPostprocessConfig | None = None,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits.detach())
    if postprocess_config is None or not postprocess_config.enabled:
        return (probabilities > threshold).to(dtype=logits.dtype)

    probabilities_np = probabilities.squeeze(1).cpu().numpy()
    processed_masks = np.stack(
        [
            filter_connected_components(
                prob_map=prob_map,
                threshold=threshold,
                config=postprocess_config,
            )
            for prob_map in probabilities_np
        ],
        axis=0,
    )

    return torch.as_tensor(
        processed_masks[:, None, :, :],
        device=logits.device,
        dtype=logits.dtype,
    )
